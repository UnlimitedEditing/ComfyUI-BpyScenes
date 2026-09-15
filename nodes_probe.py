"""Render-engine probe: headless OpenGL (moderngl) vs headless Chromium + three.js.

Both render the same visualizer-like test scene with a full post chain (bloom,
depth of field, film grain; the OpenGL one also light shafts) at the target
resolution, stepping the clock manually, and stream PROBE_GL / PROBE_WEB result
lines into the job log as they're measured. See probe/gl_probe.py and
probe/web_probe.py.
"""
import asyncio
import os
import subprocess
import sys

from .bpy_runtime import WORK_DIR, run_subprocess

_PROBE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe")


class BpyScenesRenderProbe:
    """Outputs a side-by-side video: OpenGL/moderngl (left) vs three.js (right),
    or whichever one produced frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "gl_frames":          ("INT", {"default": 3600, "min": 60, "max": 21600}),
            "web_render_frames":  ("INT", {"default": 1200, "min": 60, "max": 21600}),
            "web_capture_frames": ("INT", {"default": 1200, "min": 60, "max": 21600}),
            "width":              ("INT", {"default": 1280, "min": 64, "max": 3840}),
            "height":             ("INT", {"default": 720, "min": 64, "max": 2160}),
            "fps":                ("INT", {"default": 60, "min": 1, "max": 120}),
        }}

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("comparison", "report")
    FUNCTION     = "run"
    CATEGORY     = "BpyScenes"
    OUTPUT_NODE  = True

    async def run(self, gl_frames, web_render_frames, web_capture_frames, width, height, fps):
        from comfy_api.latest import InputImpl

        os.makedirs(WORK_DIR, exist_ok=True)
        lines = []

        def log(line):
            print(f"[BpyScenesRenderProbe] {line}", flush=True)
            lines.append(line)

        gpu = await asyncio.to_thread(subprocess.run, ["nvidia-smi", "--query-gpu=name,driver_version",
                                                       "--format=csv,noheader"], capture_output=True, text=True)
        log(f"gpu: {gpu.stdout.strip()}")

        gl_out = os.path.join(WORK_DIR, "probe_gl.mp4")
        web_out = os.path.join(WORK_DIR, "probe_web.mp4")
        for p in (gl_out, web_out):
            if os.path.isfile(p):
                os.remove(p)

        rc, out, err = await run_subprocess(
            [sys.executable, os.path.join(_PROBE_DIR, "gl_probe.py"), gl_out, str(gl_frames), str(width), str(height),
             str(fps)], stream_prefix="probe_gl")
        lines.append(out)
        if rc != 0:
            log(f"OpenGL probe failed rc={rc}: {err[-1500:]}")

        rc, out, err = await run_subprocess(
            [sys.executable, os.path.join(_PROBE_DIR, "web_probe.py"), web_out, str(web_render_frames),
             str(web_capture_frames), str(width), str(height), str(fps)], stream_prefix="probe_web")
        lines.append(out)
        if rc != 0:
            log(f"three.js probe failed rc={rc}: {err[-1500:]}")

        have = [p for p in (gl_out, web_out) if os.path.isfile(p)]
        if not have:
            raise RuntimeError("neither probe produced a video; see PROBE_GL / PROBE_WEB lines in the log")
        result = have[0]
        if len(have) == 2:
            cmp_path = os.path.join(WORK_DIR, "probe_comparison.mp4")
            cmp = await asyncio.to_thread(subprocess.run, [
                "ffmpeg", "-y", "-v", "error", "-i", gl_out, "-i", web_out,
                "-filter_complex", "[0][1]hstack=inputs=2", "-shortest",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", cmp_path,
            ], capture_output=True, text=True)
            log(f"comparison video (left OpenGL, right three.js) ok={cmp.returncode == 0}"
                + ("" if cmp.returncode == 0 else f" err={cmp.stderr[-300:]}"))
            if cmp.returncode == 0:
                result = cmp_path
        report = "\n".join(lines)
        return {"ui": {"text": [report]}, "result": (InputImpl.VideoFromFile(result), report)}
