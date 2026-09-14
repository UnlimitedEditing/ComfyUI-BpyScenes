# ComfyUI-BpyScenes

Headless Blender (EEVEE/CYCLES) scene rendering as ComfyUI nodes, built for ephemeral
cloud GPU runners such as Graydient. The nodes generate procedural, audio-reactive 3D
scenes and render them to video. No diffusion is involved.

## How Blender runs

`bpy` pip wheels exist for CPython 3.11 and 3.13+, but not 3.12, which many ComfyUI
installs use. So `bpy` is never imported into ComfyUI's venv. On first use,
`bpy_runtime.py` provisions a portable Python 3.13
([python-build-standalone](https://github.com/astral-sh/python-build-standalone)) and
pip-installs `bpy` into it. Scene scripts in `bpy_scripts/` then run as async
subprocesses, so the ComfyUI event loop is never blocked. PNG frames are muxed to MP4
with a system `ffmpeg`, because the `bpy` wheel ships without FFmpeg output.

Requires Linux x86_64, `ffmpeg` on PATH, and an NVIDIA GPU. EEVEE falls back to CPU
without one.

## Nodes

| Node | Purpose |
|---|---|
| `BpyScenesResolveAudioPath` | Audio URL / staged filename / mangled Telegram reference → local path (first non-empty of three inputs) |
| `BpyScenesAudioAnalyze` | librosa analysis → JSON: `duration`, `bpm`, `energy_timeline`, `beat_times`, `section_times` |
| `BpyScenesMusicVisualizer` | Analysis + audio → audio-reactive render with the song muxed in |
| `BpyScenesProceduralField` | Self-contained ripple-field demo render (`frame_step=2` renders on twos) |
| `BpyScenesRenderTest` | Timing diagnostic: renders a glTF sample and reports per-phase cost |
| `BpyScenesRenderBench` | Benchmarks render settings, the ESRGAN upscale stage and encoders on the job's GPU |

## Music visualizer presets

Presets come in two independent vocabularies, so any scene works with any look (36
combinations):

| Scenes (`bpy_scripts/scenes.py`) | Looks (`bpy_scripts/looks.py`) |
|---|---|
| `ripple_field`: sphere field, bass ripples, beat ring pulses | `neon_night` |
| `monolith_grid`: pillars fire on beats, slabs on the horizon | `ember` |
| `tunnel`: flight through twisting hex rings | `ice` |
| `orbital_core`: breathing core, wire cage, shard orbits | `acid` |
| `spectrum_street`: towers replay recent audio toward a horizon sun | `mono_red` |
| `helix`: rotating double helix with climbing beat pulses | `sunset` |

- **Scene** covers geometry, motion, staging, camera, and audio response.
- **Look** covers palette, accent colour, emission, key/rim lights, sky gradient, and
  distance fade.

Composition comes from staging, not compositor effects. `bpy_scripts/kit.py` provides
depth layers (floor grid, halo, dust shell) and a `CameraRig` with per-scene shot lists
(lens, off-centre lens shift, roll). The rig hard-cuts at section changes, or on the next
beat after about 7s without a cut.

Three contracts keep scenes and looks independent:

- **signals** (`signals.py`): the analysis JSON becomes per-frame arrays before
  rendering: smoothed `energy`, `low`/`mid`/`high` bands, `onset`, beat envelope,
  sections, and camera cuts. Scenes only read those arrays. Analysis without `bands` still
  works (every band follows energy).
- **drive**: each scene writes `obj.color[0]` (0–1 excitement) and `obj.color[1]`
  (0 = palette, 1 = accent colour). Looks read only those two values.
- **no colours in scenes**: scenes never set materials or colours.

`quality` picks the render path:

- `1080p_esrgan` (default) and `1440p_esrgan` render every frame at 640×360 with fast
  EEVEE settings (16 samples, no shadows, uncompressed PNG). While Blender is still
  rendering, the node upscales each frame on the GPU with Real-ESRGAN
  `realesr-animevideov3` (2.5 MB, loaded via spandrel) and pipes it straight into
  ffmpeg with the audio (NVENC when available, else x264). See `upscale.py`.
  - Upscaling runs in fp16 and falls back to fp32 if it detects non-finite output.
  - The model is expected in `models/upscale_models/` and downloaded on a cache miss.
- `720p_full` renders natively with Blender defaults. `720p_twos` and `1080p_twos` hold
  every other frame.

Frames are never skipped to save time. If the estimated cost exceeds `render_budget_s`,
the clip is shortened and the log says so (`WARNING render budget`). `max_frames` = 0
renders the whole song from `start_seconds`.

### Adding a preset

- **Look:** add an entry to `LOOKS` in `looks.py` and its name to `LOOKS` in
  `nodes_bpy.py`.
- **Scene:** add a class with `build(scn, mat, sig, intensity)` and `update(f)` to
  `scenes.py`, stage it with `kit.py`, register it in `SCENES`, and add its name to
  `SCENES` in `nodes_bpy.py`.

Scripts can be tested outside ComfyUI with any Python that has `bpy`:

```
python bpy_scripts/music_visualizer.py config.json
```

See `BpyScenesMusicVisualizer.run` for the config keys.
