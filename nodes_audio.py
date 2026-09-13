"""Audio input + analysis nodes feeding the Blender scenes.

BpyScenesResolveAudioPath  Graydient audio field -> local file path.
BpyScenesAudioAnalyze      local audio -> analysis JSON (duration, bpm,
                           energy_timeline, beat_times, section_times).
"""
import json
import os
import re
import subprocess
import tempfile
import urllib.request

# Ported from UnlimitedEditing/comfy-audio-duration's load_audio_any.py (itself
# from ComfyUI-HiggsV3Glue): Graydient can hand back a Telegram reply reference
# with every ':' and '/' stripped out.
_MANGLED_TELEGRAM_FILE_RE = re.compile(
    r'^(?:[a-z_]+__)?(https?)api\.telegram\.orgfilebot(\d+)([A-Za-z0-9_-]+?)'
    r'(voice|photo|video_note|video|audio|document|animation|sticker)file(\d+)\.([a-z0-9]+)$',
    re.IGNORECASE,
)


def _unmangle_telegram_file_url(value):
    m = _MANGLED_TELEGRAM_FILE_RE.match(value)
    if not m:
        return None
    scheme, bot_id, bot_secret, media_kind, file_id, ext = m.groups()
    return f"{scheme}://api.telegram.org/file/bot{bot_id}:{bot_secret}/{media_kind}/file_{file_id}.{ext}"


def _download(url):
    suffix = os.path.splitext(url.split("?")[0])[1] or ".audio"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(path, "wb") as f:
        f.write(resp.read())
    return path


def _resolve_to_path(value, label):
    value = (value or "").strip()
    if not value:
        return None
    if value.startswith(("http://", "https://")):
        print(f"[BpyScenesResolveAudioPath] {label}={value!r} -> URL")
        return _download(value)
    reconstructed = _unmangle_telegram_file_url(value)
    if reconstructed is not None:
        print(f"[BpyScenesResolveAudioPath] {label}={value!r} -> reconstructed Telegram URL")
        return _download(reconstructed)
    candidates = [value]
    try:
        import folder_paths
        candidates.append(os.path.join(folder_paths.get_input_directory(), value))
    except Exception:
        pass
    for candidate in candidates:
        if os.path.isfile(candidate):
            print(f"[BpyScenesResolveAudioPath] {label}={value!r} -> local file {candidate!r}")
            return candidate
    raise ValueError(f"{label}={value!r} is not a URL, a Telegram file reference, or an existing file. Tried: {candidates!r}")


class BpyScenesResolveAudioPath:
    """Three STRING inputs, first non-empty wins -- it's unpredictable which of
    init_audio_url / init_audio / init_audio_filename a Graydient submission
    path populates, so map all three."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio_source":          ("STRING", {"default": ""}),
            "audio_source_alt":      ("STRING", {"default": ""}),
            "audio_source_filename": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION     = "run"
    CATEGORY     = "BpyScenes"

    async def run(self, audio_source="", audio_source_alt="", audio_source_filename=""):
        import asyncio
        for value, label in ((audio_source, "audio_source"),
                             (audio_source_alt, "audio_source_alt"),
                             (audio_source_filename, "audio_source_filename")):
            path = await asyncio.to_thread(_resolve_to_path, value, label)
            if path:
                return (path,)
        raise RuntimeError("BpyScenesResolveAudioPath: all three audio inputs were empty.")


def _decode_mono(path, sr):
    """Decode with ffmpeg rather than librosa's soundfile/audioread backends,
    so any container ffmpeg reads (mp3, m4a, Telegram .oga opus) works."""
    import numpy as np
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}: {proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, dtype="<i2").astype("float32") / 32768.0


def analyze_audio(path):
    import librosa
    import numpy as np

    sr, hop = 22050, 512
    y = _decode_mono(path, sr)
    duration = len(y) / sr

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_max = float(rms.max()) or 1.0
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    rec = librosa.segment.recurrence_matrix(mfcc, mode="affinity", sym=True)
    seg_frames = librosa.segment.agglomerative(rec, k=min(8, rec.shape[0] - 1))
    section_times = sorted({
        round(float(t), 2) for t in librosa.frames_to_time(seg_frames, sr=sr, hop_length=hop)
        if 0 < float(t) < duration
    })

    return {
        "duration": round(duration, 2),
        "bpm": round(float(np.asarray(tempo).item()), 1),
        "energy_timeline": [
            {"time": round(float(t), 2), "energy": round(float(e) / rms_max, 3)}
            for t, e in zip(rms_times, rms) if float(t) <= duration
        ],
        "beat_times": [round(float(t), 3) for t in beat_times],
        "section_times": section_times,
    }


class BpyScenesAudioAnalyze:
    """Same analysis and JSON shape as ComfyUI-TripoSG's AudioAnalyze, without
    the spectrogram render (no matplotlib dependency, less job time)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio_path": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("analysis_json",)
    FUNCTION     = "run"
    CATEGORY     = "BpyScenes"

    async def run(self, audio_path):
        import asyncio
        import time
        t0 = time.time()
        # Tens of seconds on a full song -- keep it off the event loop.
        analysis = await asyncio.to_thread(analyze_audio, audio_path)
        print(f"[BpyScenesAudioAnalyze] {analysis['duration']}s, {analysis['bpm']} bpm, "
              f"{len(analysis['beat_times'])} beats, {len(analysis['section_times'])} section boundaries "
              f"in {time.time() - t0:.1f}s")
        return (json.dumps(analysis),)
