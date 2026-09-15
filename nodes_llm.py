"""Describe-a-visualizer: natural language -> closed-vocabulary spec.

BpyScenesSpecFromPrompt runs a local instruct model (Qwen2.5, transformers) with
generation constrained token-by-token to glviz/vocab.build_schema() via
xgrammar -- the model can only emit enums, 0-100 integers and short ids that the
schema allows. The output is then passed through vocab.normalise() (references,
relationships, rules, budget), with one retry and a preset fallback, so this
node never fails a job and always returns a spec that renders.

The constrained-decoding approach (torch_native bitmask backend, no Triton) is
ported from ComfyUI-MeshScript's IR generation nodes.

Weights: pre-staged by concept_mapping into models/llm/<model>/ (see the
workflow); falls back to a Hugging Face download on a cache miss.
"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "glviz"))

import vocab  # noqa: E402

MODELS = {"Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct"}

GUIDANCE = """You design music visualizers by writing a JSON spec in a fixed vocabulary.
Pick things from the vocabulary only. Think like a visual artist and a DJ:

- Start from the description's mood, genre and imagery. Choose 2-5 objects: exactly one or two MAIN subjects
  (particle_field, pillar_grid, ring_tunnel, hero_sphere, wire_cage, shard_swarm, tower_rows, double_helix),
  plus optional framing/backdrop (halo_ring, sun_disc, dust, grid_floor). Fewer, well-chosen objects look better.
- Map the music to motion the way a listener feels it: bass/kick -> big mass, scale, height; mid -> spin, flow,
  twist; treble/hi-hats -> sparkle, twinkle; beat/downbeat -> pulses and fired hits; onset -> flashes;
  energy -> speed; section_change -> big swells at drops.
- Use 2-6 bindings. Use punchy curves for hits, smooth or s_curve for swells, tight smoothing for percussion,
  loose for pads and ambience.
- Relationships are optional: orbit small things around a central subject, mirror for symmetry, chain or
  pulse_together to link objects. Only reference ids you defined in objects.
- Camera: fpv only with ring_tunnel or tower_rows; cinematic for variety; tele or crane for epic/ambient;
  low_dolly for dense grids. Set focus to the main subject's id.
- Look: pick the palette that matches the mood. glow 0-100 (calm 40-60, club/neon 70-90), exposure 50-70,
  light_rays higher for dramatic/spiritual, depth_of_field higher for intimate/dreamy.
- Every number is an integer 0-100. Ids are short lowercase names like core, field, rings.
- Fill every field of every item. For relationships that don't use a parameter, still give it a middle value
  (radius 50, speed 40, lag 30, symmetry radial_3) and set anchor to the main subject id.

VOCABULARY
"""


_EXAMPLE_BINDINGS = {
    "orbital_core": [
        {"source": "bass", "object": "core", "channel": "scale", "amount": 70, "curve": "smooth", "smoothing": "loose"},
        {"source": "treble", "object": "shards", "channel": "sparkle", "amount": 80, "curve": "linear",
         "smoothing": "tight"},
        {"source": "section_change", "object": "halo", "channel": "glow", "amount": 90, "curve": "s_curve",
         "smoothing": "medium"},
    ],
    "spectrum_street": [
        {"source": "bass", "object": "street", "channel": "height_left", "amount": 90, "curve": "punchy",
         "smoothing": "tight"},
        {"source": "treble", "object": "street", "channel": "height_right", "amount": 85, "curve": "linear",
         "smoothing": "tight"},
        {"source": "energy", "object": "sun", "channel": "glow", "amount": 70, "curve": "smooth", "smoothing": "loose"},
    ],
}


def _examples():
    ex = []
    for prompt, name in (("a calm icy sphere with crystals drifting around it", "orbital_core"),
                         ("synthwave drive down a neon street at sunset", "spectrum_street")):
        spec = vocab.normalise({**vocab.PRESETS[name], "bindings": _EXAMPLE_BINDINGS[name]}, log=None)
        ex.append(f'Description: "{prompt}"\nSpec: ' + json.dumps(_to_llm_shape(spec), separators=(",", ":")))
    return "\n\n".join(ex)


def _to_llm_shape(spec):
    """Resolved spec -> the exact shape the schema demands (all fields present)."""
    rels = []
    main = spec["camera"]["focus"]
    for r in spec["relationships"]:
        rels.append({"verb": r["verb"], "subjects": r["subjects"], "anchor": r.get("anchor") or main,
                     "radius": r["radius"], "speed": r["speed"], "lag": r["lag"], "symmetry": r["symmetry"]})
    return {"objects": spec["objects"], "bindings": spec["bindings"], "relationships": rels,
            "camera": spec["camera"], "look": spec["look"]}


def system_prompt():
    return GUIDANCE + vocab.vocabulary_text() + "\n\nEXAMPLES\n" + _examples() + \
        "\n\nRespond with the JSON spec only."


def _schema_variants():
    """xgrammar versions differ in JSON Schema keyword support; fall back to
    simpler schemas rather than failing (normalise() enforces the same limits)."""
    full = vocab.build_schema()
    yield "full", full

    def strip(node, keys):
        if isinstance(node, dict):
            return {k: strip(v, keys) for k, v in node.items() if k not in keys}
        if isinstance(node, list):
            return [strip(v, keys) for v in node]
        return node
    yield "no_pattern", strip(full, {"pattern"})
    yield "no_pattern_minmax", strip(full, {"pattern", "minimum", "maximum", "minItems", "maxItems"})


def _load(model_name, log):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = MODELS[model_name]
    local = None
    try:
        import folder_paths
        cand = os.path.join(folder_paths.models_dir, "llm", model_name)
        if os.path.isfile(os.path.join(cand, "config.json")):
            local = cand
        cache_dir = os.path.join(folder_paths.models_dir, "llm", "hf_cache")
    except Exception:
        cache_dir = None
    source = local or repo
    log(f"loading {model_name} from {'pre-staged ' + local if local else 'Hugging Face (not pre-staged)'}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=cache_dir)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    # `dtype=` is the newer transformers keyword, `torch_dtype=` the older one;
    # ComfyUI environments vary, so try both rather than pinning transformers.
    try:
        model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype, device_map="auto", cache_dir=cache_dir)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype, device_map="auto", cache_dir=cache_dir)
    model.eval()
    log(f"model loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def _constrained_generate(model, tokenizer, messages, temperature, seed, max_tokens, log):
    import torch
    import xgrammar as xgr
    from xgrammar.contrib.hf import LogitsProcessor

    class TorchNativeProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if len(self.matchers) == 0:
                self.batch_size = input_ids.shape[0]
                self.compiled_grammars = (self.compiled_grammars if len(self.compiled_grammars) > 1
                                          else self.compiled_grammars * self.batch_size)
                self.matchers = [xgr.GrammarMatcher(self.compiled_grammars[i]) for i in range(self.batch_size)]
                self.token_bitmask = xgr.allocate_token_bitmask(self.batch_size, self.full_vocab_size)
            if not self.prefilled:
                self.prefilled = True
            else:
                for i in range(self.batch_size):
                    if not self.matchers[i].is_terminated():
                        assert self.matchers[i].accept_token(input_ids[i][-1].item())
            for i in range(self.batch_size):
                if not self.matchers[i].is_terminated():
                    self.matchers[i].fill_next_token_bitmask(self.token_bitmask, i)
            xgr.apply_token_bitmask_inplace(scores, self.token_bitmask.to(scores.device), backend="torch_native")
            return scores

    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=model.config.vocab_size)
    compiler = xgr.GrammarCompiler(info)
    grammar = None
    for label, schema in _schema_variants():
        try:
            # Compact JSON only: free whitespace roughly doubles token count.
            try:
                grammar = compiler.compile_json_schema(json.dumps(schema), any_whitespace=False)
            except TypeError:
                grammar = compiler.compile_json_schema(json.dumps(schema))
            log(f"grammar compiled ({label} schema)")
            break
        except Exception as e:  # noqa: BLE001
            log(f"grammar compile failed for {label} schema: {str(e)[:200]}")
    if grammar is None:
        raise RuntimeError("could not compile any schema variant with xgrammar")

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    kwargs = dict(max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                  logits_processor=[TorchNativeProcessor(grammar)])
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        kwargs["do_sample"] = False
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    new_ids = out[0][inputs.input_ids.shape[1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True)
    log(f"generated {len(new_ids)} tokens in {time.time() - t0:.1f}s (prompt {inputs.input_ids.shape[1]} tokens)"
        + (" -- hit max tokens" if len(new_ids) >= max_tokens else ""))
    return raw


def spec_from_prompt(prompt, fallback_preset, model_name, temperature, seed, retries, log):
    prompt = (prompt or "").strip()
    if not prompt:
        log(f"empty description; using preset {fallback_preset}")
        return vocab.normalise(vocab.PRESETS[fallback_preset], log=log), "preset"

    model = tokenizer = None
    try:
        model, tokenizer = _load(model_name, log)
        hint = vocab.palette_hint(prompt)
        user = f'Description: "{prompt}"'
        if hint:
            user += f"\n(The description's mood words suggest the {hint} palette.)"
            log(f"palette hint from description: {hint}")
        messages = [{"role": "system", "content": system_prompt()},
                    {"role": "user", "content": user + "\nSpec:"}]
        for attempt in range(retries + 1):
            try:
                raw = _constrained_generate(model, tokenizer, messages, temperature, seed + attempt, 1600, log)
                log("raw spec: " + raw.replace("\n", " ")[:3000])
                parsed = json.loads(raw)
                spec = vocab.normalise(parsed, log=log)
                return spec, "llm"
            except Exception as e:  # noqa: BLE001
                log(f"attempt {attempt + 1} failed: {str(e)[:300]}")
    except Exception as e:  # noqa: BLE001
        log(f"LLM unavailable: {str(e)[:400]}")
    finally:
        if model is not None:
            del model, tokenizer
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    log(f"falling back to preset {fallback_preset}")
    return vocab.normalise(vocab.PRESETS[fallback_preset], log=log), "preset_fallback"


class BpyScenesSpecFromPrompt:
    """Plain-language description -> visualizer spec JSON (feed it to the
    Music Visualizer's spec_json input). Empty description = the preset."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "description": ("STRING", {"multiline": True, "default": ""}),
            "fallback_preset": (list(vocab.PRESETS), {"default": "ripple_field"}),
            "palette": (["auto", *vocab.PALETTES], {"default": "auto"}),
            "model": (list(MODELS), {"default": "Qwen2.5-7B-Instruct"}),
            "creativity": ("INT", {"default": 60, "min": 0, "max": 100}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31 - 1}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("spec_json",)
    FUNCTION = "run"
    CATEGORY = "BpyScenes"

    async def run(self, description, fallback_preset, palette, model, creativity, seed):
        import asyncio

        def log(line):
            print(f"[BpyScenesSpecFromPrompt] {line}", flush=True)

        # creativity 0-100 -> sampling temperature 0.2-1.0 (never fully greedy:
        # the same description should still be able to yield variations by seed)
        temperature = 0.2 + 0.8 * min(max(creativity, 0), 100) / 100.0
        t0 = time.time()
        spec, origin = await asyncio.to_thread(spec_from_prompt, description, fallback_preset, model, temperature,
                                               seed, 1, log)
        if palette != "auto" and palette != spec["look"]["palette"]:
            log(f"palette override: {spec['look']['palette']} -> {palette}")
            spec["look"]["palette"] = palette
        log(f"spec ready from {origin} in {time.time() - t0:.1f}s: "
            f"{[o['type'] for o in spec['objects']]} camera={spec['camera']['style']} "
            f"palette={spec['look']['palette']}")
        log("resolved spec: " + json.dumps(spec, separators=(",", ":")))
        return (json.dumps(spec),)
