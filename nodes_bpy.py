"""Blender render nodes. Scene code lives in bpy_scripts/ and runs in the
isolated Python 3.13 + bpy runtime (see bpy_runtime.py)."""
import asyncio
import json
import os
import shutil
import threading
import time
import urllib.request

from .bpy_runtime import WORK_DIR, ensure_bpy_python, run_subprocess, script_path

# Kept in sync with SCENES in bpy_scripts/scenes.py and LOOKS in bpy_scripts/looks.py.
SCENES = ["ripple_field", "monolith_grid", "tunnel", "orbital_core", "spectrum_street", "helix"]
LOOKS = ["neon_night", "ember", "ice", "acid", "mono_red", "sunset"]

# render      Blender resolution
# output      final video size; set together with upscale=True for the
#             render-low / ESRGAN-upscale path (see upscale.py)
# step        frame_step (the *_twos tiers hold every other frame)
# pf_render   est. render seconds per rendered frame
# pf_post     est. seconds per output frame after rendering (upscale + encode)
# settings    Blender render knobs (see apply_render_settings)
#
# Measured EEVEE on Graydient RTX 5090s with Blender defaults: 720p 0.136-0.142s,
# 1080p ~0.21-0.26s. The esrgan tiers' costs are estimates until bench numbers
# exist (360p render with 16 samples/no shadows, compact ESRGAN, NVENC/x264).
QUALITY = {
    "1080p_esrgan": {"render": (640, 360), "output": (1920, 1080), "upscale": True, "step": 1,
                     "pf_render": 0.05, "pf_post": 0.03,
                     "settings": {"samples": 16, "shadows": False, "image_format": "PNG", "png_compression": 0}},
    "1440p_esrgan": {"render": (640, 360), "output": (2560, 1440), "upscale": True, "step": 1,
                     "pf_render": 0.05, "pf_post": 0.045,
                     "settings": {"samples": 16, "shadows": False, "image_format": "PNG", "png_compression": 0}},
    "720p_full":    {"render": (1280, 720), "step": 1, "pf_render": 0.16, "pf_post": 0.012, "settings": {}},
    "720p_twos":    {"render": (1280, 720), "step": 2, "pf_render": 0.16, "pf_post": 0.012, "settings": {}},
    "1080p_twos":   {"render": (1920, 1080), "step": 2, "pf_render": 0.28, "pf_post": 0.027, "settings": {}},
}


class _Report:
    def __init__(self, prefix):
        self.prefix, self.lines = prefix, []

    def __call__(self, line):
        print(f"[{self.prefix}] {line}")
        self.lines.append(line)

    def text(self):
        return "\n".join(self.lines)


class BpyScenesRenderTest:
    """Timing/diagnostic node: renders a glTF sample with EEVEE or CYCLES and
    reports per-phase timing (runtime provisioning, glTF import, render, mux)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "gltf_url":    ("STRING", {"default": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Fox/glTF-Binary/Fox.glb"}),
            "width":       ("INT", {"default": 1920, "min": 64, "max": 3840}),
            "height":      ("INT", {"default": 1080, "min": 64, "max": 2160}),
            "frame_count": ("INT", {"default": 48, "min": 1, "max": 4096}),
            "fps":         ("INT", {"default": 24, "min": 1, "max": 60}),
            "engine":      (["BLENDER_EEVEE", "CYCLES"], {"default": "BLENDER_EEVEE"}),
            "samples":     ("INT", {"default": 32, "min": 1, "max": 4096}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION     = "run"
    CATEGORY     = "BpyScenes"
    OUTPUT_NODE  = True

    async def run(self, gltf_url, width, height, frame_count, fps, engine, samples):
        import asyncio
        log = _Report("BpyScenesRenderTest")
        py_bin = await ensure_bpy_python(log)

        gltf_path = os.path.join(WORK_DIR, "asset.glb")
        t0 = time.time()
        await asyncio.to_thread(urllib.request.urlretrieve, gltf_url, gltf_path)
        log(f"gltf_download_time_s: {time.time() - t0:.2f}")

        rc, stdout, stderr = await run_subprocess([
            py_bin, script_path("render_test.py"), gltf_path, os.path.join(WORK_DIR, "render_test.mp4"),
            str(width), str(height), str(frame_count), str(fps), engine, str(samples),
        ])
        log("----- subprocess stdout -----")
        log(stdout)
        if rc != 0:
            log("----- subprocess stderr -----")
            log(stderr)
        return {"ui": {"text": [log.text()]}, "result": (log.text(),)}


class BpyScenesProceduralField:
    """Self-contained procedural demo: a grid of instanced spheres carrying a
    traveling ripple under an orbiting camera. frame_step=2 renders on twos."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width":       ("INT", {"default": 1280, "min": 64, "max": 3840}),
            "height":      ("INT", {"default": 720, "min": 64, "max": 2160}),
            "frame_count": ("INT", {"default": 240, "min": 1, "max": 4096}),
            "fps":         ("INT", {"default": 24, "min": 1, "max": 60}),
            "grid_n":      ("INT", {"default": 16, "min": 2, "max": 64}),
            "frame_step":  ("INT", {"default": 1, "min": 1, "max": 4}),
        }}

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "report")
    FUNCTION     = "run"
    CATEGORY     = "BpyScenes"
    OUTPUT_NODE  = True

    async def run(self, width, height, frame_count, fps, grid_n, frame_step):
        from comfy_api.latest import InputImpl
        log = _Report("BpyScenesProceduralField")
        py_bin = await ensure_bpy_python(log)

        out_path = os.path.join(WORK_DIR, "procedural_field.mp4")
        rc, stdout, stderr = await run_subprocess([
            py_bin, script_path("procedural_field.py"), out_path,
            str(width), str(height), str(frame_count), str(fps), str(grid_n), str(frame_step),
        ])
        log("----- subprocess stdout -----")
        log(stdout)
        if rc != 0 or not os.path.isfile(out_path):
            log("----- subprocess stderr -----")
            log(stderr)
            raise RuntimeError(f"procedural_field.py failed (rc={rc}):\n{stderr[-3000:]}")
        return {"ui": {"text": [log.text()]}, "result": (InputImpl.VideoFromFile(out_path), log.text())}


class BpyScenesMusicVisualizer:
    """Audio-reactive render. Scene and look presets are independent: any scene
    works with any look. Output covers the song from start_seconds to the end
    (max_frames > 0 caps it). If the estimated render time (from measured
    per-frame costs) exceeds render_budget_s the clip is shortened with a
    WARNING in the log -- frames are never skipped to save time."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "analysis_json":   ("STRING", {"forceInput": True}),
            "audio_path":      ("STRING", {"forceInput": True}),
            "scene":           (SCENES, {"default": SCENES[0]}),
            "look":            (LOOKS, {"default": LOOKS[0]}),
            "quality":         (list(QUALITY), {"default": "1080p_esrgan"}),
            "intensity":       ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "fps":             ("INT", {"default": 24, "min": 12, "max": 60}),
            "max_frames":      ("INT", {"default": 0, "min": 0, "max": 21600}),
            "start_seconds":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.5}),
            "render_budget_s": ("INT", {"default": 150, "min": 10, "max": 1800}),
        }}

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "report")
    FUNCTION     = "run"
    CATEGORY     = "BpyScenes"
    OUTPUT_NODE  = True

    @classmethod
    def plan_frames(cls, duration, start_seconds, fps, max_frames, tier, budget_s, log):
        """Returns (output frames, frame_step)."""
        base_step, pf_render, pf_mux = tier["step"], tier["pf_render"], tier["pf_post"]
        frames = int(max(0.0, duration - start_seconds) * fps)
        if frames < 1:
            raise ValueError(f"start_seconds={start_seconds} is past the end of the song ({duration:.1f}s)")
        if 0 < max_frames < frames:
            log(f"max_frames={max_frames} caps the clip at {max_frames / fps:.1f}s of {frames / fps:.1f}s")
            frames = max_frames

        def cost(n, step):
            return len(range(0, n, step)) * pf_render + n * pf_mux

        # Never skip frames to save time: held frames visibly miss audio events.
        # If the budget can't cover the song, shorten -- loudly.
        step = base_step
        if cost(frames, step) > budget_s:
            fit = frames
            while fit > 1 and cost(fit, step) > budget_s:
                fit -= max(1, fps // 4)
            log(f"WARNING render budget: CLIP SHORTENED {frames / fps:.1f}s -> {fit / fps:.1f}s -- est "
                f"{cost(frames, step):.0f}s exceeds render_budget_s={budget_s}. Lower fps or quality to get "
                f"the whole song.")
            frames = fit
        log(f"song {duration:.1f}s: {frames} output frames ({frames / fps:.1f}s @ {fps} fps) from "
            f"{start_seconds:.1f}s, rendering {len(range(0, frames, step))} (frame_step {step}), "
            f"est {cost(frames, step):.0f}s")
        return frames, step

    async def run(self, analysis_json, audio_path, scene, look, quality, intensity,
                  fps, max_frames, start_seconds, render_budget_s):
        from comfy_api.latest import InputImpl
        log = _Report("BpyScenesMusicVisualizer")

        analysis = json.loads(analysis_json)
        tier = QUALITY[quality]
        width, height = tier["render"]
        frames, step = self.plan_frames(float(analysis.get("duration") or 0.0), start_seconds, fps, max_frames,
                                        tier, render_budget_s, log)

        py_bin = await ensure_bpy_python(log)

        os.makedirs(WORK_DIR, exist_ok=True)
        out_path = os.path.join(WORK_DIR, "music_visualizer.mp4")
        frames_dir = out_path + "_frames"
        # A warm container may hold a previous job's frames; the upscaler must
        # never pick those up before Blender clears the directory itself.
        shutil.rmtree(frames_dir, ignore_errors=True)
        cfg_path = os.path.join(WORK_DIR, "music_visualizer.json")
        with open(cfg_path, "w") as f:
            json.dump({
                "analysis": analysis, "audio_path": audio_path,
                "frames_dir": frames_dir, "out_path": out_path,
                "width": width, "height": height, "fps": fps,
                "frame_count": frames, "frame_step": step, "start_seconds": start_seconds,
                "scene": scene, "look": look, "intensity": intensity,
                "frames_only": bool(tier.get("upscale")),
                **tier["settings"],
            }, f)
        blender_cmd = [py_bin, script_path("music_visualizer.py"), cfg_path]

        t_job = time.time()
        if tier.get("upscale"):
            await self._render_and_upscale(blender_cmd, frames_dir, frames, fps, tier["output"],
                                           audio_path, start_seconds, out_path, log)
        else:
            rc, stdout, stderr = await run_subprocess(blender_cmd, stream_prefix="music_visualizer")
            log.lines.append(stdout)
            if rc != 0:
                log("----- subprocess stderr -----")
                log(stderr)
                raise RuntimeError(f"music_visualizer.py failed (rc={rc}):\n{stderr[-3000:]}")
        log(f"render+post wall time: {time.time() - t_job:.1f}s for {frames} frames "
            f"({(time.time() - t_job) / frames:.4f}s per output frame)")
        return {"ui": {"text": [log.text()]}, "result": (InputImpl.VideoFromFile(out_path), log.text())}

    @staticmethod
    async def _render_and_upscale(blender_cmd, frames_dir, frames, fps, output_size, audio_path, start_seconds,
                                  out_path, log):
        """Blender renders small frames while a worker thread upscales and encodes
        them as they land, so both GPU stages run concurrently."""
        from .upscale import FrameSink, Upscaler, pick_encoder, upscale_stream

        upscaler = await asyncio.to_thread(Upscaler, log)
        encoder = await asyncio.to_thread(pick_encoder, log)
        audio = (audio_path, start_seconds, frames / fps) if audio_path else None
        sink = FrameSink(out_path, output_size, fps, encoder, audio=audio)
        render_done = threading.Event()

        async def render():
            try:
                return await run_subprocess(blender_cmd, stream_prefix="music_visualizer")
            finally:
                render_done.set()

        render_task = asyncio.create_task(render())
        try:
            await asyncio.to_thread(upscale_stream, frames_dir, "png", frames, render_done, upscaler, sink,
                                    output_size, log)
        except BaseException:
            render_task.cancel()
            sink.kill()
            if render_done.is_set() and render_task.done() and not render_task.cancelled():
                rc, _, stderr = render_task.result()
                if rc != 0:
                    raise RuntimeError(f"music_visualizer.py failed (rc={rc}):\n{stderr[-3000:]}")
            raise
        rc, stdout, stderr = await render_task
        log.lines.append(stdout)
        if rc != 0:
            sink.kill()
            raise RuntimeError(f"music_visualizer.py failed (rc={rc}):\n{stderr[-3000:]}")
        enc_rc, enc_err = await asyncio.to_thread(sink.close)
        if enc_rc != 0:
            raise RuntimeError(f"ffmpeg ({encoder}) failed (rc={enc_rc}):\n{enc_err}")


class BpyScenesRenderBench:
    """Benchmarks visualizer render settings (EEVEE samples, shadows, PNG vs
    JPEG frames, 360p/720p/1080p), the ESRGAN upscale stage, and encoders on
    the actual job hardware. Each result is streamed to the job log the moment
    it's measured, so a timeout still keeps everything finished so far. Outputs
    a side-by-side video: native 1080p with Blender defaults (left) vs 360p
    rendered + ESRGAN-upscaled to 1080p (right)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames_per_variant": ("INT", {"default": 36, "min": 6, "max": 240}),
        }}

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("comparison", "report")
    FUNCTION     = "run"
    CATEGORY     = "BpyScenes"
    OUTPUT_NODE  = True

    async def run(self, frames_per_variant):
        import subprocess

        from comfy_api.latest import InputImpl
        log = _Report("BpyScenesRenderBench")
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        log(f"gpu: {gpu}")
        py_bin = await ensure_bpy_python(log)

        bench_dir = os.path.join(WORK_DIR, "bench")
        out_path = os.path.join(WORK_DIR, "bench_comparison.mp4")
        cfg_path = os.path.join(WORK_DIR, "bench.json")
        with open(cfg_path, "w") as f:
            json.dump({"work_dir": bench_dir, "frames": frames_per_variant, "variants": None, "out_path": out_path}, f)

        rc, stdout, stderr = await run_subprocess([py_bin, script_path("bench.py"), cfg_path], stream_prefix="bench")
        log.lines.append(stdout)
        if rc != 0:
            log("----- subprocess stderr -----")
            log(stderr)
            raise RuntimeError(f"bench.py failed (rc={rc}):\n{stderr[-3000:]}")

        esrgan_video = await self._bench_upscale(bench_dir, frames_per_variant, log)
        result_video = out_path
        base_dir = os.path.join(bench_dir, "base_1080_s64_shadow_png")
        if esrgan_video and os.path.isdir(base_dir):
            cmp_path = os.path.join(WORK_DIR, "bench_native_vs_esrgan.mp4")
            cmp = await asyncio.to_thread(subprocess.run, [
                "ffmpeg", "-y", "-v", "error", "-framerate", "24", "-i", os.path.join(base_dir, "frame_%04d.png"),
                "-i", esrgan_video, "-filter_complex", "[0][1]hstack=inputs=2",
                "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", cmp_path,
            ], capture_output=True, text=True)
            log(f"comparison video: left=native 1080p defaults, right=360p+ESRGAN ok={cmp.returncode == 0}"
                + ("" if cmp.returncode == 0 else f" err={cmp.stderr[-300:]}"))
            if cmp.returncode == 0:
                result_video = cmp_path
        if not os.path.isfile(result_video):
            raise RuntimeError("bench produced no comparison video")
        return {"ui": {"text": [log.text()]}, "result": (InputImpl.VideoFromFile(result_video), log.text())}

    @staticmethod
    async def _bench_upscale(bench_dir, frames, log):
        """Times upscale + encode on the 360p variants' frames. Returns the x264
        ESRGAN 1080p video of the ripple_field variant for the comparison."""
        from .upscale import FrameSink, Upscaler, pick_encoder, upscale_stream

        try:
            upscaler = await asyncio.to_thread(Upscaler, log)
        except Exception as e:
            log(f"UPSCALE_BENCH failed to load upscaler: {e!r}")
            return None
        done = threading.Event()
        done.set()
        import numpy as np
        # Warm-up so cuDNN autotuning isn't counted in the first measurement.
        await asyncio.to_thread(upscaler, np.zeros((8, 360, 640, 3), dtype=np.uint8), (1920, 1080))
        nvenc = await asyncio.to_thread(pick_encoder, log)
        encoders = ["libx264"] + (["h264_nvenc"] if nvenc == "h264_nvenc" else [])
        comparison = None
        for variant in ("360_s16_noshadow_png0", "360_s8_noshadow_png0"):
            vdir = os.path.join(bench_dir, variant)
            if not os.path.isdir(vdir):
                continue
            for size in ((1920, 1080), (2560, 1440)):
                for enc in encoders:
                    out = os.path.join(bench_dir, f"esrgan_{variant}_{size[1]}_{enc}.mp4")
                    sink = FrameSink(out, size, 24, enc)
                    try:
                        stats = await asyncio.to_thread(upscale_stream, vdir, "png", frames, done, upscaler, sink,
                                                        size, lambda line: None, 8, False, 0)
                    finally:
                        rc, err = await asyncio.to_thread(sink.close)
                    log(f"UPSCALE_BENCH {variant} -> {size[0]}x{size[1]} {enc}: ok={rc == 0} "
                        f"decode={stats['decode_s'] * 1000:.1f}ms upscale={stats['upscale_s'] * 1000:.1f}ms "
                        f"encode_write={stats['encode_write_s'] * 1000:.1f}ms "
                        f"total={stats['wall_s'] / frames * 1000:.1f}ms/frame" + ("" if rc == 0 else f" err={err[-200:]}"))
                    if rc == 0 and comparison is None and size == (1920, 1080) and enc == "libx264" \
                            and variant == "360_s16_noshadow_png0":
                        comparison = out
        return comparison
