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

## Music visualizer presets

Presets come in two independent vocabularies, so any scene works with any look:

- **scene** covers geometry, motion, camera, and audio response: `ripple_field`,
  `monolith_grid`
- **look** covers palette, emission, lighting, and world: `neon_night`, `ember`

Two contracts in `bpy_scripts/music_visualizer.py` keep them independent:

- **signals**: the analysis JSON is turned into per-frame arrays (smoothed energy,
  beat envelope, section index/progress) before rendering. Scenes only read those
  arrays.
- **drive**: each scene writes a 0–1 "excitement" value per element into
  `obj.color[0]`. Looks read only that value (Object Info → Color → R).

`quality` picks resolution and frame step (`720p_twos`, `1080p_twos`, `720p_full`). The
node lowers the frame count so that estimated render + mux time fits `render_budget_s`.

### Adding a preset

- **Look:** add an entry to `LOOKS` in the script and its name to `LOOKS` in
  `nodes_bpy.py`.
- **Scene:** add a class with `build(scn, mat, sig, intensity)` and `update(f)`, register
  it in `SCENES`, and add its name to `SCENES` in `nodes_bpy.py`. Scenes must write
  drive and must never set colours.

Scripts can be tested outside ComfyUI with any Python that has `bpy`:

```
python bpy_scripts/music_visualizer.py config.json
```

See `BpyScenesMusicVisualizer.run` for the config keys.
