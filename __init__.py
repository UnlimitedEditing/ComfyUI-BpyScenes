from .nodes_audio import BpyScenesAudioAnalyze, BpyScenesResolveAudioPath
from .nodes_llm import BpyScenesSpecFromPrompt
from .nodes_probe import BpyScenesRenderProbe
from .nodes_bpy import (BpyScenesMusicVisualizer, BpyScenesProceduralField, BpyScenesRenderBench,
                        BpyScenesRenderTest)

NODE_CLASS_MAPPINGS = {
    "BpyScenesResolveAudioPath": BpyScenesResolveAudioPath,
    "BpyScenesAudioAnalyze":     BpyScenesAudioAnalyze,
    "BpyScenesMusicVisualizer":  BpyScenesMusicVisualizer,
    "BpyScenesProceduralField":  BpyScenesProceduralField,
    "BpyScenesRenderTest":       BpyScenesRenderTest,
    "BpyScenesRenderBench":      BpyScenesRenderBench,
    "BpyScenesRenderProbe":      BpyScenesRenderProbe,
    "BpyScenesSpecFromPrompt":   BpyScenesSpecFromPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BpyScenesResolveAudioPath": "BpyScenes: Resolve Audio Source -> Path",
    "BpyScenesAudioAnalyze":     "BpyScenes: Audio Analyze (beats/energy/sections)",
    "BpyScenesMusicVisualizer":  "BpyScenes: Music Visualizer (scene x look)",
    "BpyScenesProceduralField":  "BpyScenes: Procedural Field (demo render)",
    "BpyScenesRenderTest":       "BpyScenes: Render Test (timing diagnostic)",
    "BpyScenesRenderBench":      "BpyScenes: Render Settings Benchmark",
    "BpyScenesRenderProbe":      "BpyScenes: Render Engine Probe (OpenGL vs three.js)",
    "BpyScenesSpecFromPrompt":   "BpyScenes: Describe a Visualizer (LLM -> spec)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
