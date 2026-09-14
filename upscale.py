"""Render-low, upscale-on-GPU video path.

Blender renders small frames (e.g. 640x360); this module upscales them with
Real-ESRGAN animevideov3 (SRVGGNetCompact, 2.5 MB, loaded via spandrel -- a
ComfyUI core dependency) *while Blender is still rendering*, and pipes the
result straight into ffmpeg together with the audio, so full-size frames never
touch disk.

animevideov3 was picked over realesr-general-x4v3 (dark halo ringing on hard
CG edges) after side-by-side tests; 360p -> x4 -> area-downscale to 1080p came
out close to native, 270p thickened thin lines too much.
"""
import os
import subprocess
import time
import urllib.request

from .bpy_runtime import WORK_DIR

MODEL_NAME = "realesr-animevideov3.pth"
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"


def model_path(log):
    """Prefer the concept_mapping pre-download in models/upscale_models/, else
    download once (2.5 MB)."""
    target_dir = os.path.join(WORK_DIR, "models")
    try:
        import folder_paths
        found = folder_paths.get_full_path("upscale_models", MODEL_NAME)
        if found:
            return found
        target_dir = folder_paths.get_folder_paths("upscale_models")[0]
    except Exception:
        pass
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, MODEL_NAME)
    if not os.path.isfile(path):
        log(f"upscale model not pre-staged, downloading {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class Upscaler:
    def __init__(self, log):
        import torch
        from spandrel import ModelLoader

        try:
            import comfy.model_management as mm
            self.device = mm.get_torch_device()
        except Exception:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log = log
        self.model = ModelLoader().load_from_file(model_path(log)).model.to(self.device).eval()
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model.to(self.dtype)
        log(f"upscaler: {MODEL_NAME} on {self.device} ({self.dtype})")

    def __call__(self, frames, out_size):
        """frames: uint8 numpy (B, H, W, 3) -> uint8 numpy (B, out_h, out_w, 3)."""
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(frames).to(self.device).permute(0, 3, 1, 2)
        with torch.inference_mode():
            y = self.model(x.to(self.dtype).div_(255))
            if self.dtype == torch.float16 and not torch.isfinite(y).all():
                # Some GPUs (e.g. GTX 16-series) produce NaNs in fp16 conv; they
                # show up as black frames. Switch permanently to fp32.
                self.log("upscaler: non-finite fp16 output, switching to fp32")
                self.dtype = torch.float32
                self.model.to(self.dtype)
                y = self.model(x.to(self.dtype).div_(255))
            out_w, out_h = out_size
            if y.shape[-2:] != (out_h, out_w):
                mode = "area" if y.shape[-1] > out_w else "bicubic"
                y = F.interpolate(y, size=(out_h, out_w), mode=mode)
            y = y.clamp_(0, 1).mul_(255).round_().to(torch.uint8)
            return y.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def pick_encoder(log, prefer="auto"):
    """h264_nvenc when the container's driver supports it (checked with a tiny
    real encode, since ffmpeg can list the encoder without a usable device),
    else libx264."""
    if prefer in ("libx264", "h264_nvenc"):
        return prefer
    probe = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=320x240:rate=24:duration=0.2",
                            "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, text=True)
    enc = "h264_nvenc" if probe.returncode == 0 else "libx264"
    log(f"encoder: {enc}" + ("" if enc == "h264_nvenc" else f" (nvenc unavailable: {probe.stderr.strip()[-160:]})"))
    return enc


class FrameSink:
    """ffmpeg reading raw RGB frames on stdin, optionally muxing an audio window."""

    def __init__(self, out_path, size, fps, encoder, audio=None):
        w, h = size
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-framerate", str(fps), "-i", "-"]
        if audio:
            path, start, duration = audio
            cmd += ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", path,
                    "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k"]
        if encoder == "h264_nvenc":
            cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "19", "-b:v", "0"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
        cmd += ["-pix_fmt", "yuv420p", out_path]
        self.stderr_path = out_path + ".ffmpeg.log"
        self._stderr = open(self.stderr_path, "w")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr,
                                     start_new_session=True)

    def write(self, frames):
        self.proc.stdin.write(frames.tobytes())

    def close(self):
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        rc = self.proc.wait()
        self._stderr.close()
        return rc, open(self.stderr_path).read()[-2000:]

    def kill(self):
        try:
            self.proc.kill()
        except Exception:
            pass


def _decode(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def upscale_stream(frames_dir, ext, total, render_done, upscaler, sink, out_size, log,
                   batch=8, delete=True, progress_every=240):
    """Consume frame_0001..frame_{total} as Blender writes them. A frame counts as
    complete once the next one exists (Blender writes in order) or rendering has
    finished. Blocking -- run it in a worker thread.

    render_done: threading.Event set when the Blender process has exited."""
    import numpy as np

    def path(i):
        return os.path.join(frames_dir, f"frame_{i:04d}.{ext}")

    i = 1
    t_decode = t_up = t_write = 0.0
    t_begin = time.time()
    while i <= total:
        ready = []
        while len(ready) < batch and i + len(ready) <= total:
            idx = i + len(ready)
            if os.path.isfile(path(idx)) and (os.path.isfile(path(idx + 1)) or render_done.is_set()):
                ready.append(path(idx))
            else:
                break
        if not ready:
            if render_done.is_set() and not os.path.isfile(path(i)):
                raise RuntimeError(f"render finished but frame {i}/{total} is missing")
            time.sleep(0.02)
            continue
        t0 = time.time()
        arr = np.stack([_decode(p) for p in ready])
        t1 = time.time()
        out = upscaler(arr, out_size)
        t2 = time.time()
        sink.write(out)
        t3 = time.time()
        t_decode, t_up, t_write = t_decode + t1 - t0, t_up + t2 - t1, t_write + t3 - t2
        if delete:
            for p in ready:
                os.remove(p)
        prev, i = i, i + len(ready)
        if progress_every and (i - 1) // progress_every != (prev - 1) // progress_every:
            log(f"upscale progress {i - 1}/{total} frames, elapsed {time.time() - t_begin:.1f}s")
    n = max(total, 1)
    stats = {"frames": total, "decode_s": t_decode / n, "upscale_s": t_up / n, "encode_write_s": t_write / n,
             "wall_s": time.time() - t_begin}
    log(f"upscale done: {total} frames, per frame decode {stats['decode_s'] * 1000:.1f}ms "
        f"upscale {stats['upscale_s'] * 1000:.1f}ms encode-write {stats['encode_write_s'] * 1000:.1f}ms, "
        f"wall {stats['wall_s']:.1f}s")
    return stats
