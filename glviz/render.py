"""GL music visualizer entry point (runs with ComfyUI's own Python + moderngl).

Usage: python render.py <config.json>
config keys: analysis, audio_path, out_path, width, height, supersample, fps,
             frame_count, start_seconds, scene, look, intensity, bitrate

Steps the song clock manually (t = f / fps), renders every frame on the GPU,
reads it back and pipes it to ffmpeg (NVENC when available) with the audio.
Prints progress and a per-stage COST breakdown so every run calibrates the
budget estimates.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(os.path.dirname(HERE), "bpy_scripts"))  # signals.py, looks_data.py (no bpy)

from gpu import create_context  # noqa: E402
from looks_data import LOOKS  # noqa: E402
from renderer import Renderer  # noqa: E402
from gl_scenes import SCENES  # noqa: E402
from signals import Signals  # noqa: E402


def log(line):
    print(line, flush=True)


def open_encoder(cfg, width, height, log):
    probe = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=320x240:rate=30:duration=0.2",
                            "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True)
    bitrate = cfg.get("bitrate", "10M")
    if probe.returncode == 0:
        venc = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-b:v", bitrate, "-maxrate", "16M",
                "-bufsize", "24M"]
    else:
        venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-maxrate", "16M", "-bufsize", "24M"]
    log(f"encoder: {venc[1]} target {bitrate}")
    n, fps = cfg["frame_count"], cfg["fps"]
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-framerate", str(fps), "-i", "-"]
    audio = cfg.get("audio_path")
    if audio:
        cmd += ["-ss", f"{cfg['start_seconds']:.3f}", "-t", f"{n / fps:.3f}", "-i", audio,
                "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-vf", "vflip", *venc, "-pix_fmt", "yuv420p", "-movflags", "+faststart", cfg["out_path"]]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    cfg = json.load(open(sys.argv[1]))
    t_start = time.time()
    fps, n = cfg["fps"], cfg["frame_count"]
    ow, oh = cfg["width"], cfg["height"]
    ss = max(1, int(cfg.get("supersample", 1)))

    sig = Signals(cfg["analysis"], n, fps, cfg["start_seconds"])
    ctx = create_context(log)
    r = Renderer(ctx, (ow * ss, oh * ss), (ow, oh), LOOKS[cfg["look"]])
    scene = SCENES[cfg["scene"]]()
    scene.build(r, sig, cfg["intensity"])
    enc = open_encoder(cfg, ow, oh, log)
    t_setup = time.time() - t_start
    log(f"setup {t_setup:.2f}s: {cfg['scene']} / {cfg['look']}, render {ow * ss}x{oh * ss} -> output {ow}x{oh}, "
        f"{n} frames @ {fps} fps ({n / fps:.1f}s)")

    t_update = t_gpu = t_read = t_write = 0.0
    t_loop = time.time()
    every = max(1, n // 10)
    for f in range(n):
        a = time.time()
        scene.update(f)
        b = time.time()
        r.draw(f)
        r.finish()
        c = time.time()
        data = r.read()
        d = time.time()
        enc.stdin.write(data)
        e = time.time()
        t_update, t_gpu, t_read, t_write = t_update + b - a, t_gpu + c - b, t_read + d - c, t_write + e - d
        if f == 0 or (f + 1) % every == 0:
            el = time.time() - t_loop
            log(f"progress {f + 1}/{n} elapsed {el:.1f}s eta {el / (f + 1) * (n - f - 1):.1f}s")
    enc.stdin.close()
    rc = enc.wait()
    err = enc.stderr.read().decode(errors="replace")
    wall = time.time() - t_loop
    ms = 1000.0 / max(n, 1)
    log(f"COST gl {ow}x{oh} ss{ss}: scene_update={t_update * ms:.2f}ms gpu_render+post={t_gpu * ms:.2f}ms "
        f"readback={t_read * ms:.2f}ms encode_write={t_write * ms:.2f}ms total={wall * ms:.2f}ms/frame "
        f"({n / wall:.0f} fps, {n / fps / wall:.1f}x realtime) wall={wall:.1f}s setup={t_setup:.1f}s")
    if rc != 0:
        log(f"ffmpeg failed rc={rc}: {err[-1500:]}")
    sys.exit(0 if rc == 0 else 3)


if __name__ == "__main__":
    main()
