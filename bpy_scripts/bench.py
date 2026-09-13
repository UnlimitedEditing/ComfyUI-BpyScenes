"""Render-settings benchmark for the music visualizer.

Usage: python bench.py <config.json>
config: {"work_dir": str, "frames": int, "variants": [names] | null, "out_path": str}

Renders the real visualizer scenes (synthetic 128 bpm signals, no audio needed)
under a series of settings variants, printing one RESULT line per variant as
soon as it finishes -- so if the job is killed by the platform timeout, every
variant that completed is already in the log. Then times the encode paths
(x264 at 1080p, lanczos 720p->1080p upscale, NVENC if present) and writes a
side-by-side comparison video (baseline | fastest 1080p variant).
"""
import json
import math
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import music_visualizer as mv  # noqa: E402

FPS = 24

# (name, scene, look, width, height, settings). Ordered by how much each answer
# matters, so a timeout still leaves the important rows.
VARIANTS = [
    ("base_1080_s64_shadow_png", "ripple_field", "neon_night", 1920, 1080, {}),
    ("1080_s16_shadow_png",      "ripple_field", "neon_night", 1920, 1080, {"samples": 16}),
    ("1080_s8_noshadow_png",     "ripple_field", "neon_night", 1920, 1080, {"samples": 8, "shadows": False}),
    ("1080_s8_noshadow_jpg",     "ripple_field", "neon_night", 1920, 1080, {"samples": 8, "shadows": False, "image_format": "JPEG"}),
    ("1080_s4_noshadow_jpg",     "ripple_field", "neon_night", 1920, 1080, {"samples": 4, "shadows": False, "image_format": "JPEG"}),
    ("720_s8_noshadow_jpg",      "ripple_field", "neon_night", 1280, 720,  {"samples": 8, "shadows": False, "image_format": "JPEG"}),
    ("mono_1080_s8_noshadow_jpg", "monolith_grid", "ember",    1920, 1080, {"samples": 8, "shadows": False, "image_format": "JPEG"}),
    ("mono_1080_s8_shadow_jpg",  "monolith_grid", "ember",     1920, 1080, {"samples": 8, "shadows": True, "image_format": "JPEG"}),
    ("mono_1080_s64_shadow_png", "monolith_grid", "ember",     1920, 1080, {}),
]


def synthetic_analysis(seconds):
    beat = 60.0 / 128
    return {
        "duration": seconds,
        "bpm": 128.0,
        "beat_times": [round(i * beat, 3) for i in range(int(seconds / beat) + 1)],
        "energy_timeline": [{"time": round(t * 0.05, 2), "energy": round(0.55 + 0.45 * math.sin(t * 0.05 * 1.7), 3)}
                            for t in range(int(seconds / 0.05) + 1)],
        "section_times": [round(seconds / 2, 2)],
    }


def run_variant(name, scene, look, width, height, settings, frames, work_dir):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.app.handlers.frame_change_pre.clear()
    scn = bpy.context.scene
    t0 = time.time()
    sig = mv.Signals(synthetic_analysis(frames / FPS + 1), frames, FPS, 0.0)
    mat = mv.build_look(look)
    impl = mv.SCENES[scene]()
    impl.build(scn, mat, sig, 1.0)
    bpy.app.handlers.frame_change_pre.append(lambda s: impl.update(min(max(s.frame_current - 1, 0), frames - 1)))
    cfg = {"width": width, "height": height, **settings}
    mv.apply_render_settings(scn, cfg)
    scn.render.fps = FPS
    scn.frame_start, scn.frame_end, scn.frame_step = 1, frames, 1
    frames_dir = os.path.join(work_dir, name)
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)
    scn.render.filepath = frames_dir + "/frame_"
    t_setup = time.time() - t0

    timer = mv.FrameTimer(frames)
    t0 = time.time()
    bpy.ops.render.render(animation=True)
    t_wall = time.time() - t0
    timer.detach()
    mv.log(f"RESULT {name}: {scene}/{look} {width}x{height} {mv.render_settings_summary(scn, cfg)} "
           f"setup={t_setup:.2f}s wall={t_wall:.2f}s wall_per_frame={t_wall / frames:.4f}s {timer.summary()}")
    return frames_dir, mv.image_ext(scn), t_wall / frames


def time_encode(label, pattern, out_path, frames, **kw):
    t0 = time.time()
    r = mv.encode_video(pattern, FPS, out_path, FPS, frames, **kw)
    dt = time.time() - t0
    ok = r.returncode == 0
    mv.log(f"ENCODE {label}: ok={ok} total={dt:.2f}s per_frame={dt / frames:.4f}s"
           + ("" if ok else f" err={r.stderr.strip()[-300:]}"))
    return ok


def main():
    cfg = json.load(open(sys.argv[1]))
    work_dir, frames = cfg["work_dir"], cfg["frames"]
    wanted = cfg.get("variants")
    t_start = time.time()
    mv.log(f"bench: {frames} frames per variant, bpy {bpy.app.version_string}")

    done = {}
    for v in VARIANTS:
        if wanted and v[0] not in wanted:
            continue
        done[v[0]] = run_variant(*v, frames, work_dir)
        mv.log(f"bench elapsed {time.time() - t_start:.1f}s")

    if "1080_s8_noshadow_jpg" in done:
        d, ext, _ = done["1080_s8_noshadow_jpg"]
        time_encode("x264_veryfast_1080_from_jpg", f"{d}/frame_%04d.{ext}", os.path.join(work_dir, "enc_x264.mp4"), frames)
        time_encode("nvenc_1080_from_jpg", f"{d}/frame_%04d.{ext}", os.path.join(work_dir, "enc_nvenc.mp4"), frames,
                    encoder="h264_nvenc")
    if "720_s8_noshadow_jpg" in done:
        d, ext, _ = done["720_s8_noshadow_jpg"]
        time_encode("x264_lanczos_720_to_1080", f"{d}/frame_%04d.{ext}", os.path.join(work_dir, "enc_up.mp4"), frames,
                    out_size=(1920, 1080))

    # Side-by-side: baseline (left) vs fastest 1080p ripple variant (right).
    fast = [k for k in ("1080_s4_noshadow_jpg", "1080_s8_noshadow_jpg") if k in done]
    if "base_1080_s64_shadow_png" in done and fast:
        (bd, be, _), (fd, fe, _) = done["base_1080_s64_shadow_png"], done[fast[0]]
        cmp = subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-framerate", str(FPS), "-i", f"{bd}/frame_%04d.{be}",
            "-framerate", str(FPS), "-i", f"{fd}/frame_%04d.{fe}",
            "-filter_complex", "[0][1]hstack=inputs=2",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", cfg["out_path"],
        ], capture_output=True, text=True)
        mv.log(f"comparison video: left=base_1080_s64_shadow_png right={fast[0]} ok={cmp.returncode == 0}"
               + ("" if cmp.returncode == 0 else f" err={cmp.stderr[-300:]}"))
    mv.log(f"bench total {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
