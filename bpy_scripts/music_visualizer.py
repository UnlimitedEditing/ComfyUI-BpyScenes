"""Music visualizer entry point.

Runs inside the portable Python 3.13 + pip bpy subprocess launched by
BpyScenesMusicVisualizer -- never imported into ComfyUI's own venv (bpy has no
3.12 wheel). Usage: python music_visualizer.py <config.json>

Presets are two independent vocabularies (see README):
  scenes.py  SCENES  geometry, motion, staging, camera, audio response
  looks.py   LOOKS   palette, emission, lighting, world
held apart by the signals contract (signals.py) and the drive contract
(obj.color[0] drive / obj.color[1] accent, see looks.py). Shared staging and
camera tools live in kit.py.
"""
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402

from looks import LOOKS, build_look  # noqa: E402,F401
from scenes import SCENES  # noqa: E402,F401
from signals import Signals  # noqa: E402,F401


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
