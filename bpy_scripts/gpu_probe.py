"""Which GPU does EEVEE actually render on, and what does a visualizer frame cost?

Usage: python gpu_probe.py <work_dir> <frames>

Graydient hosts can expose an AMD iGPU (Mesa) alongside the NVIDIA card, and
EGL may default to the iGPU; only CUDA/Cycles had been confirmed on NVIDIA.
Renders the real ripple_field scene at 1280x720 (16 samples, no shadows, PNG
level 0) and prints BLENDER_GPU lines with gpu.platform's vendor/renderer and
the steady per-frame cost. Run it with and without __EGL_VENDOR_LIBRARY_FILENAMES
pointing at the NVIDIA vendor file to compare.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench  # noqa: E402
import gpu  # noqa: E402


def main():
    work_dir, frames = sys.argv[1], int(sys.argv[2])
    label = os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES", "default EGL vendor")
    settings = {"samples": 16, "shadows": False, "image_format": "PNG", "png_compression": 0}
    _, _, per_frame = bench.run_variant("gpu_probe_720", "ripple_field", "neon_night", 1280, 720, settings, frames,
                                        work_dir)
    try:
        info = (f"vendor={gpu.platform.vendor_get()!r} renderer={gpu.platform.renderer_get()!r} "
                f"version={gpu.platform.version_get()!r} backend={gpu.platform.backend_type_get()!r}")
    except Exception as e:  # noqa: BLE001
        info = f"gpu.platform unavailable: {e}"
    print(f"BLENDER_GPU [{label}] {info} wall_per_frame_720p_s16={per_frame:.4f}s", flush=True)


if __name__ == "__main__":
    main()
