"""Blender render nodes. Scene code lives in bpy_scripts/ and runs in the
isolated Python 3.13 + bpy runtime (see bpy_runtime.py)."""
import json
import os
import time
import urllib.request

from .bpy_runtime import WORK_DIR, ensure_bpy_python, run_subprocess, script_path

# Kept in sync with SCENES in bpy_scripts/scenes.py and LOOKS in bpy_scripts/looks.py.
SCENES = ["ripple_field", "monolith_grid", "tunnel", "orbital_core", "spectrum_street", "helix"]
LOOKS = ["neon_night", "ember", "ice", "acid", "mono_red", "sunset"]

# (width, height, frame_step, render s per rendered frame, mux s per output frame).
# Measured EEVEE on Graydient RTX 5090s: 720p 0.136s, 1080p ~0.21-0.26s, and
# 2K mux 11.3s/240 frames scaled by pixel count -- rounded up for headroom.
QUALITY = {
    "720p_twos":  (1280, 720, 2, 0.16, 0.012),
    "1080p_twos": (1920, 1080, 2, 0.28, 0.027),
    "720p_full":  (1280, 720, 1, 0.16, 0.012),
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
    works with any look. Output length follows the song from start_seconds,
    capped by max_frames and then by render_budget_s (estimated from measured
    per-frame costs) so a long song can't run the job out of time."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "analysis_json":   ("STRING", {"forceInput": True}),
            "audio_path":      ("STRING", {"forceInput": True}),
            "scene":           (SCENES, {"default": SCENES[0]}),
            "look":            (LOOKS, {"default": LOOKS[0]}),
            "quality":         (list(QUALITY), {"default": "720p_twos"}),
            "intensity":       ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "fps":             ("INT", {"default": 24, "min": 12, "max": 60}),
            "max_frames":      ("INT", {"default": 1440, "min": 24, "max": 4320}),
            "start_seconds":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.5}),
            "render_budget_s": ("INT", {"default": 150, "min": 10, "max": 1800}),
        }}

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "report")
    FUNCTION     = "run"
    CATEGORY     = "BpyScenes"
    OUTPUT_NODE  = True

    async def run(self, analysis_json, audio_path, scene, look, quality, intensity,
                  fps, max_frames, start_seconds, render_budget_s):
        from comfy_api.latest import InputImpl
        log = _Report("BpyScenesMusicVisualizer")

        analysis = json.loads(analysis_json)
        width, height, step, pf_render, pf_mux = QUALITY[quality]
        duration = float(analysis.get("duration") or 0.0)
        frames = min(max_frames, int(max(0.0, duration - start_seconds) * fps))
        if frames < 1:
            raise ValueError(f"start_seconds={start_seconds} is past the end of the song ({duration:.1f}s)")
        per_out = pf_render / step + pf_mux
        budget_frames = int(render_budget_s / per_out)
        if frames > budget_frames:
            log(f"clamped {frames} -> {budget_frames} frames to fit render_budget_s={render_budget_s} "
                f"(est {per_out:.3f}s per output frame at {quality})")
            frames = budget_frames
        log(f"song {duration:.1f}s, rendering {frames} frames ({frames / fps:.1f}s) from {start_seconds:.1f}s, "
            f"est {frames * per_out:.0f}s")

        py_bin = await ensure_bpy_python(log)

        out_path = os.path.join(WORK_DIR, "music_visualizer.mp4")
        cfg_path = os.path.join(WORK_DIR, "music_visualizer.json")
        with open(cfg_path, "w") as f:
            json.dump({
                "analysis": analysis, "audio_path": audio_path,
                "frames_dir": out_path + "_frames", "out_path": out_path,
                "width": width, "height": height, "fps": fps,
                "frame_count": frames, "frame_step": step, "start_seconds": start_seconds,
                "scene": scene, "look": look, "intensity": intensity,
            }, f)

        rc, stdout, stderr = await run_subprocess([py_bin, script_path("music_visualizer.py"), cfg_path],
                                                  stream_prefix="music_visualizer")
        log.lines.append(stdout)
        if rc != 0:
            log("----- subprocess stderr -----")
            log(stderr)
            raise RuntimeError(f"music_visualizer.py failed (rc={rc}):\n{stderr[-3000:]}")
        return {"ui": {"text": [log.text()]}, "result": (InputImpl.VideoFromFile(out_path), log.text())}


class BpyScenesRenderBench:
    """Benchmarks visualizer render settings (EEVEE samples, shadows, PNG vs
    JPEG frames, 720p vs 1080p) and encode paths on the actual job hardware.
    Each result is streamed to the job log the moment it's measured, so a
    timeout still keeps everything finished so far. Outputs a side-by-side
    video: default settings (left) vs the fastest 1080p variant (right)."""

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
        if rc != 0 or not os.path.isfile(out_path):
            log("----- subprocess stderr -----")
            log(stderr)
            raise RuntimeError(f"bench.py failed (rc={rc}):\n{stderr[-3000:]}")
        return {"ui": {"text": [log.text()]}, "result": (InputImpl.VideoFromFile(out_path), log.text())}
