"""Music visualizer scene assembly.

Runs inside the portable Python 3.13 + pip bpy subprocess launched by
BpyMusicVisualizer -- never imported into ComfyUI's own venv (bpy has no 3.12
wheel). Usage: python music_visualizer.py <config.json>

Presets are two independent vocabularies, held apart by two contracts:

  signals  audio analysis -> per-output-frame arrays (energy, beat envelope,
           section index/progress), computed once before rendering. Scenes only
           index into them, so frame_step skipping can't change the animation.
  drive    every scene writes a 0-1 "excitement" scalar per element into
           obj.color[0]; looks only read that (Object Info > Color > Red). A
           look never knows which scene it is on, and vice versa.

SCENES own geometry, motion, camera and audio response (these are coupled).
LOOKS own palette, emission and world (these only depend on drive).
"""
import bisect
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time

import bpy


def clamp01(v):
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def smoothstep(v):
    v = clamp01(v)
    return v * v * (3 - 2 * v)


def interp(xs, ys, x):
    if not xs:
        return 0.0
    i = bisect.bisect_right(xs, x)
    if i <= 0:
        return ys[0]
    if i >= len(xs):
        return ys[-1]
    x0, x1 = xs[i - 1], xs[i]
    return ys[i - 1] + (ys[i] - ys[i - 1]) * ((x - x0) / (x1 - x0) if x1 > x0 else 0.0)


# ── signals contract ─────────────────────────────────────────────────────────

class Signals:
    """Per-output-frame arrays. Index f is 0-based; t[f] is song time in seconds."""

    ATTACK_S = 0.03
    RELEASE_S = 0.25
    BEAT_DECAY_S = 0.15
    MIN_SECTION_S = 2.0

    def __init__(self, analysis, frame_count, fps, start_seconds):
        n = frame_count
        dt = 1.0 / fps
        self.fps = fps
        self.t = [start_seconds + f * dt for f in range(n)]

        timeline = analysis.get("energy_timeline") or []
        et = [p["time"] for p in timeline]
        ev = [p["energy"] for p in timeline]
        raw = [interp(et, ev, t) for t in self.t]
        # Normalise against the rendered window, not the whole song, so a quiet
        # intro excerpt still gets full motion range.
        ref = sorted(raw)[int(0.95 * (n - 1))] if n else 1.0
        ref = ref if ref > 1e-6 else 1.0
        k_att = 1 - math.exp(-dt / self.ATTACK_S)
        k_rel = 1 - math.exp(-dt / self.RELEASE_S)
        self.energy, s = [], 0.0
        for v in raw:
            v = min(1.0, v / ref)
            s += (v - s) * (k_att if v > s else k_rel)
            self.energy.append(s)

        beats = sorted(analysis.get("beat_times") or [])
        self.beat, self.beat_index, self.since_beat = [], [], []
        for t in self.t:
            i = bisect.bisect_right(beats, t) - 1
            if i < 0:
                self.beat.append(0.0)
                self.beat_index.append(-1)
                self.since_beat.append(1e9)
            else:
                since = t - beats[i]
                self.beat.append(math.exp(-since / self.BEAT_DECAY_S))
                self.beat_index.append(i)
                self.since_beat.append(since)

        duration = float(analysis.get("duration") or (self.t[-1] if self.t else 0.0))
        # librosa's agglomerative segmentation emits near-duplicate boundaries
        # (e.g. 18.62, 18.72) -- merge them or every section-driven camera move
        # restarts several times in a fraction of a second.
        bounds = [0.0]
        for b in sorted(analysis.get("section_times") or []):
            if b - bounds[-1] >= self.MIN_SECTION_S:
                bounds.append(b)
        bounds.append(max(duration, self.t[-1] + dt if n else 0.0))
        self.section, self.section_t, self.section_elapsed = [], [], []
        for t in self.t:
            i = min(max(bisect.bisect_right(bounds, t) - 1, 0), len(bounds) - 2)
            lo, hi = bounds[i], bounds[i + 1]
            self.section.append(i)
            self.section_t.append(clamp01((t - lo) / (hi - lo)) if hi > lo else 0.0)
            self.section_elapsed.append(t - lo)

    def cumulative(self, rate_fn):
        """Integrate a per-frame rate (units/second) so speed can follow energy
        without the position jumping back when energy drops."""
        out, acc = [], 0.0
        for f in range(len(self.t)):
            out.append(acc)
            acc += rate_fn(f) / self.fps
        return out


# ── looks (read drive only) ──────────────────────────────────────────────────

# ramp: drive -> colour. strength: drive -> emission (low end near zero so calm
# elements read as lit surfaces, not flat glow). sun: key light energy so shape
# survives when emission is low. view: colour management transform -- AgX (the
# Blender default) desaturates bright emission into pastels.
LOOKS = {
    "neon_night": {
        "ramp": [(0.0, (0.01, 0.02, 0.25)), (0.5, (0.05, 0.3, 1.0)), (1.0, (1.0, 0.3, 0.05))],
        "strength": (0.05, 4.0),
        "world": (0.004, 0.005, 0.01),
        "sun": 2.5,
        "view": "Standard",
    },
    "ember": {
        "ramp": [(0.0, (0.05, 0.01, 0.005)), (0.45, (0.55, 0.06, 0.01)), (1.0, (1.0, 0.8, 0.35))],
        "strength": (0.02, 5.0),
        "world": (0.006, 0.003, 0.002),
        "sun": 2.0,
        "view": "Standard",
    },
}


def build_look(name):
    spec = LOOKS[name]
    mat = bpy.data.materials.new(f"look_{name}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    info = nt.nodes.new("ShaderNodeObjectInfo")
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    elems = ramp.color_ramp.elements
    (p0, c0), (p1, c1) = spec["ramp"][0], spec["ramp"][-1]
    elems[0].position, elems[0].color = p0, (*c0, 1)
    elems[1].position, elems[1].color = p1, (*c1, 1)
    for pos, col in spec["ramp"][1:-1]:
        e = elems.new(pos)
        e.color = (*col, 1)
    strength = nt.nodes.new("ShaderNodeMapRange")
    strength.inputs["To Min"].default_value = spec["strength"][0]
    strength.inputs["To Max"].default_value = spec["strength"][1]
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.45
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(info.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Red"], ramp.inputs["Fac"])
    nt.links.new(sep.outputs["Red"], strength.inputs["Value"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Emission Color"])
    nt.links.new(strength.outputs["Result"], bsdf.inputs["Emission Strength"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    scn = bpy.context.scene
    world = bpy.data.worlds.new(f"world_{name}")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (*spec["world"], 1)
    scn.world = world
    sun = bpy.data.objects.new("look_sun", bpy.data.lights.new("look_sun", type="SUN"))
    sun.data.energy = spec["sun"]
    sun.rotation_euler = (math.radians(50), 0, math.radians(35))
    scn.collection.objects.link(sun)
    scn.view_settings.view_transform = spec["view"]
    return mat


# ── scenes (write drive, never touch colours) ────────────────────────────────

def add_camera(scn, target_z=0.0):
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    scn.collection.objects.link(cam)
    target = bpy.data.objects.new("Target", None)
    target.location = (0, 0, target_z)
    scn.collection.objects.link(target)
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    scn.camera = cam
    return cam


def section_eased(sig, f, values, ease_s=1.2):
    """Value for the current section, eased in from the previous section's."""
    s = sig.section[f]
    cur, prev = values[s % len(values)], values[(s - 1) % len(values)] if s > 0 else values[0]
    return prev + (cur - prev) * smoothstep(sig.section_elapsed[f] / ease_s)


class RippleField:
    """Grid of spheres carrying a traveling ripple whose height and speed follow
    energy; every beat launches a ring pulse from the centre. Orbiting camera
    changes elevation per section."""

    GRID_N, SPACING, WAVELENGTH = 16, 0.6, 6.0
    PULSE_SPEED, PULSE_WIDTH = 7.0, 0.7

    def build(self, scn, mat, sig, intensity):
        self.sig, self.k = sig, intensity
        mesh = bpy.data.meshes.new("sphere")
        import bmesh
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.18)
        bm.to_mesh(mesh)
        bm.free()
        mesh.materials.append(mat)
        half = (self.GRID_N - 1) / 2.0
        self.objs = []
        for ix in range(self.GRID_N):
            for iy in range(self.GRID_N):
                x, y = (ix - half) * self.SPACING, (iy - half) * self.SPACING
                o = bpy.data.objects.new(f"sph_{ix}_{iy}", mesh)
                o.location = (x, y, 0)
                scn.collection.objects.link(o)
                self.objs.append((o, math.hypot(x, y)))
        self.phase = sig.cumulative(lambda f: 0.35 + 0.9 * sig.energy[f] * intensity)
        self.orbit = sig.cumulative(lambda f: (2 * math.pi / 24.0) * (0.6 + 0.8 * sig.energy[f]))
        self.orbit_empty = bpy.data.objects.new("Orbit", None)
        scn.collection.objects.link(self.orbit_empty)
        self.cam = add_camera(scn)
        self.cam.parent = self.orbit_empty

    def update(self, f):
        sig, k = self.sig, self.k
        amp = 0.15 + 1.0 * sig.energy[f] * k
        pulse_r = sig.since_beat[f] * self.PULSE_SPEED
        pulse_h = 0.9 * k * sig.beat[f]
        span = amp + pulse_h + 1e-6
        for o, dist in self.objs:
            z = amp * math.sin(2 * math.pi * (self.phase[f] - dist / self.WAVELENGTH))
            z += pulse_h * math.exp(-((dist - pulse_r) / self.PULSE_WIDTH) ** 2)
            o.location.z = z
            drive = clamp01(0.5 + 0.5 * z / span)
            s = 0.6 + 0.6 * drive
            o.scale = (s, s, s)
            o.color = (drive, 0.0, 0.0, 1.0)
        self.orbit_empty.rotation_euler.z = self.orbit[f]
        self.cam.location = (14, 0, section_eased(sig, f, [9.0, 4.5, 12.0, 6.5]))


class MonolithGrid:
    """Grid of pillars: energy sets the shared base height, each beat fires a
    deterministic random subset that jumps and decays. Low camera drifts around
    the grid and swings to a new side each section."""

    GRID_N, SPACING, FIRE_P = 10, 1.1, 0.22

    def build(self, scn, mat, sig, intensity):
        self.sig, self.k = sig, intensity
        mesh = bpy.data.meshes.new("pillar")
        w = 0.4
        verts = [(-w, -w, 0), (w, -w, 0), (w, w, 0), (-w, w, 0),
                 (-w, -w, 1), (w, -w, 1), (w, w, 1), (-w, w, 1)]
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        mesh.from_pydata(verts, [], faces)
        mesh.materials.append(mat)
        half = (self.GRID_N - 1) / 2.0
        rng = random.Random(7)
        self.objs = []
        for ix in range(self.GRID_N):
            for iy in range(self.GRID_N):
                o = bpy.data.objects.new(f"pil_{ix}_{iy}", mesh)
                o.location = ((ix - half) * self.SPACING, (iy - half) * self.SPACING, 0)
                scn.collection.objects.link(o)
                self.objs.append((o, ix * self.GRID_N + iy, rng.uniform(0.6, 1.8), rng.uniform(0, 6.28)))
        self.drift = sig.cumulative(lambda f: 0.05 + 0.1 * sig.energy[f])
        self.cam = add_camera(scn, target_z=0.4)

    def fired(self, beat_index, i):
        return beat_index >= 0 and random.Random(beat_index * 1009 + i).random() < self.FIRE_P

    def update(self, f):
        sig, k = self.sig, self.k
        t, e, b, bi = sig.t[f], sig.energy[f], sig.beat[f], sig.beat_index[f]
        max_h = 0.15 + 1.0 * k + 3.0 * k
        for o, i, w, ph in self.objs:
            h = 0.15 + 1.0 * k * e * (0.2 + 0.8 * (0.5 + 0.5 * math.sin(t * w + ph)))
            if self.fired(bi, i):
                h += 3.0 * k * b
            o.scale = (1.0, 1.0, max(h, 0.02))
            o.color = (clamp01(h / max_h), 0.0, 0.0, 1.0)
        a = section_eased(sig, f, [0.6, 2.5, 4.4, 1.5]) + self.drift[f]
        self.cam.location = (15 * math.cos(a), 15 * math.sin(a), section_eased(sig, f, [7.0, 4.0, 9.0, 5.5]))


SCENES = {"ripple_field": RippleField, "monolith_grid": MonolithGrid}


# ── render settings, timing, encode (shared with bench.py) ───────────────────

def log(line):
    # Piped stdout is block-buffered; flush so the ComfyUI node can stream it.
    print(line, flush=True)


def apply_render_settings(scn, cfg):
    """Performance knobs, all optional in cfg. Defaults are Blender's own, so a
    cfg without them renders exactly as before these knobs existed."""
    scn.render.engine = "BLENDER_EEVEE"
    scn.render.resolution_x, scn.render.resolution_y = cfg["width"], cfg["height"]
    scn.render.resolution_percentage = 100
    if "samples" in cfg:
        scn.eevee.taa_render_samples = cfg["samples"]
    if "shadows" in cfg:
        scn.eevee.use_shadows = cfg["shadows"]
        for obj in scn.objects:
            if obj.type == "LIGHT":
                obj.data.use_shadow = cfg["shadows"]
    fmt = cfg.get("image_format", "PNG")
    scn.render.image_settings.file_format = fmt
    if fmt == "PNG" and "png_compression" in cfg:
        scn.render.image_settings.compression = cfg["png_compression"]
    if fmt == "JPEG":
        scn.render.image_settings.quality = cfg.get("jpeg_quality", 95)


def image_ext(scn):
    return {"PNG": "png", "JPEG": "jpg", "BMP": "bmp", "TARGA_RAW": "tga"}[scn.render.image_settings.file_format]


def render_settings_summary(scn, cfg):
    return (f"samples={scn.eevee.taa_render_samples} shadows={scn.eevee.use_shadows} "
            f"format={scn.render.image_settings.file_format} encoder={cfg.get('encoder', 'libx264')}")


class FrameTimer:
    """Per-frame time from Blender's render handlers, with periodic progress
    lines (elapsed + ETA) while the animation renders. render_post fires after
    the image is written, so each sample is render + image write together;
    compare PNG vs JPEG variants to see the write cost."""

    def __init__(self, total, progress_every=0):
        self.total, self.every = total, progress_every
        self.frame_s = []
        self.t_begin = time.time()
        self._pre = None
        self.handlers = [(bpy.app.handlers.render_pre, self.on_pre),
                         (bpy.app.handlers.render_post, self.on_post)]
        for lst, fn in self.handlers:
            lst.append(fn)

    def on_pre(self, *_):
        self._pre = time.time()

    def on_post(self, *_):
        if self._pre is None:
            return
        self.frame_s.append(time.time() - self._pre)
        done = len(self.frame_s)
        if self.every and (done % self.every == 0 or done == 1):
            elapsed = time.time() - self.t_begin
            eta = elapsed / done * (self.total - done)
            log(f"progress {done}/{self.total} frames  elapsed {elapsed:.1f}s  eta {eta:.1f}s  "
                f"last frame {self.frame_s[-1]:.3f}s")

    def detach(self):
        for lst, fn in self.handlers:
            if fn in lst:
                lst.remove(fn)

    def summary(self, skip=2):
        """The steady-state mean skips the first frames, which include EEVEE
        shader compilation."""
        xs = self.frame_s[skip:] if len(self.frame_s) > skip else self.frame_s
        steady = sum(xs) / len(xs) if xs else 0.0
        first = self.frame_s[0] if self.frame_s else 0.0
        return f"frames={len(self.frame_s)} first_frame={first:.3f}s steady_render_plus_write={steady:.4f}s"


def encode_video(pattern, in_rate, out_path, fps, frames, out_size=None,
                 extra_inputs=(), extra_output=(), encoder="libx264"):
    cmd = ["ffmpeg", "-y", "-v", "error", "-framerate", str(in_rate), "-i", pattern, *extra_inputs, *extra_output]
    if out_size:
        cmd += ["-vf", f"scale={out_size[0]}:{out_size[1]}:flags=lanczos"]
    cmd += ["-r", str(fps), "-frames:v", str(frames), "-pix_fmt", "yuv420p"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast"] if encoder == "libx264" else ["-c:v", encoder]
    cmd += [out_path]
    return subprocess.run(cmd, capture_output=True, text=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = json.load(open(sys.argv[1]))
    t_start = time.time()
    fps, n, step = cfg["fps"], cfg["frame_count"], max(1, cfg["frame_step"])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene

    sig = Signals(cfg["analysis"], n, fps, cfg["start_seconds"])
    mat = build_look(cfg["look"])
    scene_impl = SCENES[cfg["scene"]]()
    scene_impl.build(scn, mat, sig, cfg["intensity"])

    def on_frame(s):
        scene_impl.update(min(max(s.frame_current - 1, 0), n - 1))

    bpy.app.handlers.frame_change_pre.append(on_frame)

    apply_render_settings(scn, cfg)
    scn.render.fps = fps
    scn.frame_start, scn.frame_end, scn.frame_step = 1, n, step
    rendered = len(range(1, n + 1, step))

    frames_dir = os.path.abspath(cfg["frames_dir"])
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)
    scn.render.filepath = frames_dir + "/frame_"
    t_setup = time.time() - t_start
    log(f"setup done in {t_setup:.2f}s, rendering {rendered} frames at {cfg['width']}x{cfg['height']}")

    timer = FrameTimer(rendered, progress_every=max(1, rendered // 10))
    t0 = time.time()
    bpy.ops.render.render(animation=True)
    t_render = time.time() - t0
    timer.detach()

    t0 = time.time()
    ext = image_ext(scn)
    for i, f in enumerate(range(1, n + 1, step)):
        os.rename(f"{frames_dir}/frame_{f:04d}.{ext}", f"{frames_dir}/seq_{i + 1:04d}.{ext}")
    audio = cfg.get("audio_path")
    extra_in, extra_out = [], []
    if audio:
        extra_in = ["-ss", f"{cfg['start_seconds']:.3f}", "-t", f"{n / fps:.3f}", "-i", audio]
        extra_out = ["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k"]
    mux = encode_video(f"{frames_dir}/seq_%04d.{ext}", f"{fps}/{step}", cfg["out_path"], fps, n,
                       out_size=cfg.get("output_size"), extra_inputs=extra_in, extra_output=extra_out,
                       encoder=cfg.get("encoder", "libx264"))
    t_mux = time.time() - t0

    log("===== BPY MUSIC VISUALIZER RESULTS =====")
    log(f"scene: {cfg['scene']}  look: {cfg['look']}  intensity: {cfg['intensity']}")
    log(f"render_resolution: {cfg['width']}x{cfg['height']}  output_size: {cfg.get('output_size')}")
    log(f"render_settings: {render_settings_summary(scn, cfg)}")
    log(f"output_frames: {n} @ {fps}fps  frame_step: {step}  rendered_frames: {rendered}")
    log(f"song_window_s: {cfg['start_seconds']:.2f} -> {cfg['start_seconds'] + n / fps:.2f}")
    log(f"sections_in_window: {len(set(sig.section))}  beats_in_window: {len(set(b for b in sig.beat_index if b >= 0))}")
    log(f"setup_time_s: {t_setup:.2f}")
    log(f"render_time_s: {t_render:.2f}")
    log(f"frame_timing: {timer.summary()}")
    log(f"mux_time_s: {t_mux:.2f}  mux_ok: {mux.returncode == 0}")
    if mux.returncode != 0:
        log(mux.stderr)
    log(f"total_time_s: {time.time() - t_start:.2f}")
    sys.exit(0 if mux.returncode == 0 else 3)


if __name__ == "__main__":
    main()
