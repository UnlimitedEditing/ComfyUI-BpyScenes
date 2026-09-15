"""Closed vocabulary for user-built visualizers, plus normalisation, rules and presets.

A spec is only ever enums and 0-100 integers (plus short ids). This module is
the single source of truth for what's allowed: the JSON Schema the LLM is
constrained to (build_schema), the prompt vocabulary text (vocabulary_text),
and normalise(), which turns anything -- LLM output, partial or odd specs --
into a resolved spec that is guaranteed to render. Nothing here raises on bad
input: it clamps, snaps and logs.
"""
import copy
import json
import math
import re
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bpy_scripts"))
from looks_data import LOOKS  # noqa: E402

SPEC_VERSION = 1
MAX_OBJECTS, MAX_BINDINGS, MAX_RELATIONSHIPS = 7, 12, 5
MAX_MAIN_SUBJECTS = 3
INSTANCE_BUDGET = 6000
GROUNDED = {"particle_field", "pillar_grid", "tower_rows", "ring_tunnel"}

# Mood words -> palette, used as a hint to the LLM when the palette is "auto".
PALETTE_WORDS = {
    "ice": ["ice", "icy", "frozen", "cold", "arctic", "glacier", "winter", "frost", "crystal", "cool blue"],
    "ember": ["fire", "ember", "lava", "magma", "flame", "burning", "warm", "amber", "heat", "volcanic"],
    "sunset": ["sunset", "synthwave", "outrun", "retro", "dusk", "pink", "miami", "vaporwave"],
    "neon_night": ["neon", "cyberpunk", "night", "club", "blade runner", "tokyo", "electric blue"],
    "acid": ["acid", "toxic", "rave", "psychedelic", "slime", "lime", "radioactive", "green"],
    "mono_red": ["monochrome", "black and white", "noir", "minimal", "brutalist", "red and black", "stark"],
}


def palette_hint(text):
    t = (text or "").lower()
    scores = {p: sum(1 for w in words if w in t) for p, words in PALETTE_WORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None

# ── nouns ────────────────────────────────────────────────────────────────────
# channels: what audio may drive (each 0-1 internally, mapped to tuned ranges).
# defaults: bindings applied to any channel the spec leaves unbound, so no
#           object is ever dead. (source, amount, curve, smoothing)
# cost:     instance count per density. singleton: at most one per spec.
OBJECTS = {
    "particle_field": {
        "desc": "a flat grid of glowing spheres that ripples like water; bass lifts waves, beats send rings outward",
        "channels": {"height": "wave height", "wave_speed": "how fast ripples travel", "pulse": "ring pulse on hits",
                     "sparkle": "random twinkles"},
        "defaults": {"height": ("bass", 90, "smooth", "medium"), "wave_speed": ("mid", 60, "linear", "loose"),
                     "pulse": ("beat", 80, "punchy", "tight"), "sparkle": ("treble", 60, "linear", "tight")},
        "cost": {"low": 144, "medium": 324, "high": 676}, "placement": "center", "group": "main",
    },
    "pillar_grid": {
        "desc": "a city-like grid of rectangular pillars; bass raises them, beats fire random pillars upward",
        "channels": {"height": "overall pillar height", "fire": "beat-fired pillar jumps", "sparkle": "top flicker"},
        "defaults": {"height": ("bass", 80, "smooth", "medium"), "fire": ("beat", 85, "punchy", "tight"),
                     "sparkle": ("treble", 50, "linear", "tight")},
        "cost": {"low": 64, "medium": 144, "high": 256}, "placement": "center", "group": "main",
    },
    "ring_tunnel": {
        "desc": "an endless tunnel of hexagonal rings the camera can fly through; beats pulse down the tunnel",
        "channels": {"pulse": "ring pulse travelling away", "twist": "ring rotation", "speed": "flight speed",
                     "sparkle": "wall streak flicker"},
        "defaults": {"pulse": ("beat", 85, "punchy", "tight"), "twist": ("mid", 60, "smooth", "loose"),
                     "speed": ("energy", 70, "smooth", "loose"), "sparkle": ("treble", 70, "linear", "tight")},
        "cost": {"low": 60, "medium": 104, "high": 160}, "placement": "center", "group": "main", "singleton": True,
    },
    "hero_sphere": {
        "desc": "one large faceted sphere at the heart of the scene that breathes with the music",
        "channels": {"scale": "size swell", "spin": "rotation", "glow": "emission"},
        "defaults": {"scale": ("bass", 70, "punchy", "medium"), "spin": ("mid", 40, "smooth", "loose"),
                     "glow": ("energy", 70, "smooth", "medium")},
        "cost": {"low": 1, "medium": 1, "high": 1}, "placement": "center", "group": "main",
    },
    "wire_cage": {
        "desc": "a counter-rotating wireframe polyhedron cage, good wrapped around a hero sphere",
        "channels": {"scale": "size swell", "spin": "rotation", "glow": "emission"},
        "defaults": {"scale": ("mid", 40, "smooth", "medium"), "spin": ("mid", 60, "smooth", "loose"),
                     "glow": ("mid", 70, "smooth", "medium")},
        "cost": {"low": 1, "medium": 1, "high": 1}, "placement": "center", "group": "main",
    },
    "shard_swarm": {
        "desc": "hundreds of small crystal shards on tilted orbits around the centre; hi-hats make them sparkle",
        "channels": {"speed": "orbit speed", "sparkle": "twinkle", "flash": "flash on hits"},
        "defaults": {"speed": ("mid", 60, "smooth", "loose"), "sparkle": ("treble", 70, "linear", "tight"),
                     "flash": ("onset", 70, "punchy", "tight")},
        "cost": {"low": 60, "medium": 120, "high": 220}, "placement": "center", "group": "orbit",
    },
    "tower_rows": {
        "desc": "two rows of towers lining a street to the horizon; each tower replays recent audio so the music rolls away into the distance",
        "channels": {"height_left": "left row height", "height_right": "right row height",
                     "dash_pulse": "street centre-line flashes"},
        "defaults": {"height_left": ("bass", 90, "smooth", "tight"), "height_right": ("treble", 90, "linear", "tight"),
                     "dash_pulse": ("beat", 80, "punchy", "tight")},
        "cost": {"low": 72, "medium": 132, "high": 196}, "placement": "center", "group": "main", "singleton": True,
    },
    "double_helix": {
        "desc": "a rotating DNA-like double helix of spheres with rungs; beats climb up the strands",
        "channels": {"spin": "rotation speed", "pulse": "pulses climbing the strands", "sparkle": "twinkle",
                     "swell": "surrounding ring swell"},
        "defaults": {"spin": ("energy", 60, "smooth", "loose"), "pulse": ("beat", 85, "punchy", "tight"),
                     "sparkle": ("treble", 60, "linear", "tight"), "swell": ("bass", 60, "smooth", "medium")},
        "cost": {"low": 90, "medium": 148, "high": 220}, "placement": "center", "group": "main",
    },
    "halo_ring": {
        "desc": "a large thin glowing ring that frames the subject",
        "channels": {"scale": "size swell", "glow": "emission"},
        "defaults": {"scale": ("bass", 30, "smooth", "medium"), "glow": ("beat", 70, "punchy", "tight")},
        "cost": {"low": 1, "medium": 1, "high": 1}, "placement": "center", "group": "frame",
    },
    "sun_disc": {
        "desc": "a huge glowing sun disc low on the horizon, a synthwave-style focal point",
        "channels": {"scale": "size swell", "glow": "emission"},
        "defaults": {"scale": ("bass", 15, "smooth", "loose"), "glow": ("energy", 60, "smooth", "loose")},
        "cost": {"low": 1, "medium": 1, "high": 1}, "placement": "background", "group": "backdrop", "singleton": True,
    },
    "dust": {
        "desc": "sparse tiny particles drifting on a distant shell, adds depth and hi-hat sparkle",
        "channels": {"sparkle": "twinkle", "drift": "rotation of the shell"},
        "defaults": {"sparkle": ("treble", 80, "linear", "tight"), "drift": ("energy", 40, "smooth", "loose")},
        "cost": {"low": 80, "medium": 150, "high": 300}, "placement": "background", "group": "backdrop",
        "singleton": True,
    },
    "grid_floor": {
        "desc": "an infinite glowing grid floor that gives a horizon and perspective lines",
        "channels": {"glow": "grid brightness"},
        "defaults": {"glow": ("onset", 50, "punchy", "tight")},
        "cost": {"low": 0, "medium": 0, "high": 0}, "placement": "floor", "group": "backdrop", "singleton": True,
    },
}

# ── inputs ───────────────────────────────────────────────────────────────────
SOURCES = {
    "bass": "low-frequency energy (kick, sub, bassline)",
    "mid": "mid-range energy (vocals, synths, snare body)",
    "treble": "high-frequency energy (hi-hats, cymbals, air)",
    "energy": "overall loudness",
    "onset": "every transient hit",
    "beat": "the beat grid pulse",
    "downbeat": "the first beat of each bar only",
    "brightness": "spectral brightness (how bright/harsh the timbre is)",
    "section_change": "a swell at each new song section",
}
# Musicians' words for sources. Spelling-based snapping would map "hats" to
# "bass", which is exactly wrong, so these resolve first.
SOURCE_ALIASES = {
    "kick": "bass", "kicks": "bass", "sub": "bass", "subs": "bass", "low": "bass", "lows": "bass", "808": "bass",
    "bassline": "bass",
    "hats": "treble", "hat": "treble", "hihat": "treble", "hihats": "treble", "hi_hats": "treble",
    "cymbal": "treble", "cymbals": "treble", "high": "treble", "highs": "treble", "air": "treble",
    "snare": "mid", "snares": "mid", "vocal": "mid", "vocals": "mid", "mids": "mid", "synth": "mid", "chords": "mid",
    "loudness": "energy", "volume": "energy", "rms": "energy", "amplitude": "energy",
    "transient": "onset", "transients": "onset", "hit": "onset", "hits": "onset", "drums": "onset",
    "bar": "downbeat", "bars": "downbeat", "one": "downbeat",
    "timbre": "brightness", "centroid": "brightness", "spectral_centroid": "brightness",
    "section": "section_change", "sections": "section_change", "drop": "section_change",
}
CURVES = {"linear": "proportional", "s_curve": "gentle at extremes, responsive in the middle",
          "punchy": "quiet stays still, hits pop hard", "smooth": "soft and floaty"}
SMOOTHING = {"tight": "snappy", "medium": "balanced", "loose": "slow and flowing"}
DENSITIES = ["low", "medium", "high"]
PLACEMENTS = ["center", "above", "below", "left", "right", "background", "floor"]
COLOR_ROLES = ["palette", "accent"]

# ── grammar ──────────────────────────────────────────────────────────────────
VERBS = {
    "orbit": "subject circles the anchor (params: radius, speed)",
    "follow": "subject trails the anchor's movement (param: lag)",
    "mirror": "subject is duplicated symmetrically (param: symmetry)",
    "chain": "subjects react one after another with a stagger (param: lag)",
    "pulse_together": "subjects share one pulse",
}
SYMMETRIES = {"x": 2, "radial_3": 3, "radial_4": 4, "radial_6": 6}
POSITIONAL = {"orbit", "follow"}

CAMERA_STYLES = {
    "cinematic": "cuts between wide orbit, telephoto, low dolly and top-down",
    "orbit_wide": "slow wide orbit around the subject",
    "low_dolly": "low grazing push-in, big foreground",
    "crane": "high sweeping overview",
    "tele": "long-lens compressed, distant and dramatic",
    "top_down": "overhead, rotated",
    "fpv": "first-person flight (needs ring_tunnel or tower_rows)",
}
CUTS = {"on_sections": "cut on song sections", "every_8_bars": "cut every 8 bars", "never": "one continuous shot"}
PALETTES = list(LOOKS)
LOOK_KNOBS = ["glow", "exposure", "light_rays", "depth_of_field"]

ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,23}$")


# ── presets (Tier 1; also few-shot examples for the LLM) ─────────────────────

def _obj(id_, type_, size=50, density="medium", placement=None, color="palette"):
    o = {"id": id_, "type": type_, "size": size, "density": density, "color": color}
    if placement:
        o["placement"] = placement
    return o


PRESETS = {
    "ripple_field": {
        "objects": [_obj("field", "particle_field", 55), _obj("halo", "halo_ring", 55, color="accent"),
                    _obj("floor", "grid_floor"), _obj("dust", "dust", density="low")],
        "bindings": [], "relationships": [],
        "camera": {"style": "cinematic", "cuts": "on_sections", "shake": 30, "focus": "field"},
        "look": {"palette": "neon_night", "glow": 70, "exposure": 60, "light_rays": 50, "depth_of_field": 50},
    },
    "monolith_grid": {
        "objects": [_obj("pillars", "pillar_grid", 55), _obj("floor", "grid_floor"), _obj("dust", "dust", density="low")],
        "bindings": [{"source": "downbeat", "target": "pillars.fire", "amount": 90, "curve": "punchy",
                      "smoothing": "tight"}],
        "relationships": [],
        "camera": {"style": "low_dolly", "cuts": "on_sections", "shake": 35, "focus": "pillars"},
        "look": {"palette": "ember", "glow": 65, "exposure": 60, "light_rays": 60, "depth_of_field": 50},
    },
    "tunnel": {
        "objects": [_obj("tunnel", "ring_tunnel", 55)],
        "bindings": [], "relationships": [],
        "camera": {"style": "fpv", "cuts": "on_sections", "shake": 40, "focus": "tunnel"},
        "look": {"palette": "neon_night", "glow": 75, "exposure": 60, "light_rays": 40, "depth_of_field": 30},
    },
    "orbital_core": {
        "objects": [_obj("core", "hero_sphere", 55), _obj("cage", "wire_cage", 55, color="accent"),
                    _obj("shards", "shard_swarm", 55), _obj("halo", "halo_ring", 70, color="accent"),
                    _obj("dust", "dust")],
        "bindings": [], "relationships": [{"verb": "orbit", "subject": "shards", "anchor": "core", "radius": 40,
                                            "speed": 30}],
        "camera": {"style": "cinematic", "cuts": "on_sections", "shake": 25, "focus": "core"},
        "look": {"palette": "ice", "glow": 70, "exposure": 55, "light_rays": 60, "depth_of_field": 60},
    },
    "spectrum_street": {
        "objects": [_obj("street", "tower_rows", 55), _obj("sun", "sun_disc", 60), _obj("floor", "grid_floor"),
                    _obj("dust", "dust", density="low")],
        "bindings": [], "relationships": [],
        "camera": {"style": "fpv", "cuts": "on_sections", "shake": 25, "focus": "street"},
        "look": {"palette": "sunset", "glow": 70, "exposure": 60, "light_rays": 70, "depth_of_field": 40},
    },
    "helix": {
        "objects": [_obj("helix", "double_helix", 55), _obj("floor", "grid_floor"), _obj("dust", "dust")],
        "bindings": [], "relationships": [],
        "camera": {"style": "cinematic", "cuts": "on_sections", "shake": 25, "focus": "helix"},
        "look": {"palette": "acid", "glow": 70, "exposure": 60, "light_rays": 50, "depth_of_field": 50},
    },
}


# ── schema (what the LLM is token-constrained to) ────────────────────────────

def _int():
    return {"type": "integer", "minimum": 0, "maximum": 100}


def build_schema():
    obj = {"type": "object", "additionalProperties": False,
           "required": ["id", "type", "size", "density", "placement", "color"],
           "properties": {"id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,15}$"},
                          "type": {"enum": list(OBJECTS)}, "size": _int(), "density": {"enum": DENSITIES},
                          "placement": {"enum": PLACEMENTS}, "color": {"enum": COLOR_ROLES}}}
    all_channels = sorted({c for spec in OBJECTS.values() for c in spec["channels"]})
    binding = {"type": "object", "additionalProperties": False,
               "required": ["source", "object", "channel", "amount", "curve", "smoothing"],
               "properties": {"source": {"enum": list(SOURCES)}, "object": {"type": "string",
                                                                           "pattern": "^[a-z][a-z0-9_]{0,15}$"},
                              "channel": {"enum": all_channels}, "amount": _int(), "curve": {"enum": list(CURVES)},
                              "smoothing": {"enum": list(SMOOTHING)}}}
    rel = {"type": "object", "additionalProperties": False,
           "required": ["verb", "subjects", "anchor", "radius", "speed", "lag", "symmetry"],
           "properties": {"verb": {"enum": list(VERBS)},
                          "subjects": {"type": "array", "minItems": 1, "maxItems": 6,
                                       "items": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,15}$"}},
                          "anchor": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,15}$"},
                          "radius": _int(), "speed": _int(), "lag": _int(), "symmetry": {"enum": list(SYMMETRIES)}}}
    return {
        "type": "object", "additionalProperties": False,
        "required": ["objects", "bindings", "relationships", "camera", "look"],
        "properties": {
            "objects": {"type": "array", "minItems": 1, "maxItems": MAX_OBJECTS, "items": obj},
            "bindings": {"type": "array", "maxItems": MAX_BINDINGS, "items": binding},
            "relationships": {"type": "array", "maxItems": MAX_RELATIONSHIPS, "items": rel},
            "camera": {"type": "object", "additionalProperties": False,
                       "required": ["style", "cuts", "shake", "focus"],
                       "properties": {"style": {"enum": list(CAMERA_STYLES)}, "cuts": {"enum": list(CUTS)},
                                      "shake": _int(),
                                      "focus": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,15}$"}}},
            "look": {"type": "object", "additionalProperties": False,
                     "required": ["palette", *LOOK_KNOBS],
                     "properties": {"palette": {"enum": PALETTES}, **{k: _int() for k in LOOK_KNOBS}}},
        },
    }


def vocabulary_text():
    """Prompt-side description of the vocabulary, generated from the same tables
    as the schema -- adding an object or source here extends the LLM too."""
    lines = ["OBJECTS (type: what it is; channels audio can drive):"]
    for name, spec in OBJECTS.items():
        chans = ", ".join(f"{c} ({d})" for c, d in spec["channels"].items())
        lines.append(f"- {name}: {spec['desc']}. Channels: {chans}.")
    lines.append("\nAUDIO SOURCES: " + "; ".join(f"{k} = {v}" for k, v in SOURCES.items()))
    lines.append("CURVES: " + "; ".join(f"{k} = {v}" for k, v in CURVES.items()))
    lines.append("SMOOTHING: " + "; ".join(f"{k} = {v}" for k, v in SMOOTHING.items()))
    lines.append("RELATIONSHIP VERBS: " + "; ".join(f"{k} = {v}" for k, v in VERBS.items()))
    lines.append("CAMERA STYLES: " + "; ".join(f"{k} = {v}" for k, v in CAMERA_STYLES.items()))
    lines.append("CAMERA CUTS: " + "; ".join(f"{k} = {v}" for k, v in CUTS.items()))
    lines.append("PALETTES: " + ", ".join(PALETTES))
    lines.append("PLACEMENTS: " + ", ".join(PLACEMENTS) + ". DENSITY: low, medium, high. "
                 "COLOR: palette (main colours) or accent (contrasting colour).")
    return "\n".join(lines)


# ── normalisation ────────────────────────────────────────────────────────────

def _clamp_int(v, default=50):
    try:
        return int(min(max(round(float(v)), 0), 100))
    except (TypeError, ValueError):
        return default


def _snap_enum(value, allowed, default):
    """Unknown enum values snap to the closest allowed name (shared-prefix /
    character overlap), else the default."""
    if value in allowed:
        return value
    if not isinstance(value, str) or not value:
        return default
    v = value.lower().strip().replace("-", "_").replace(" ", "_")
    if v in allowed:
        return v

    def score(a):
        common = sum(1 for x, y in zip(v, a) if x == y)
        overlap = len(set(v) & set(a)) / max(len(set(v) | set(a)), 1)
        return common + overlap
    best = max(allowed, key=score)
    return best if score(best) >= 2 else default


def _clean_id(value, used, fallback):
    s = re.sub(r"[^a-z0-9_]", "_", str(value or "").lower()).strip("_")[:16] or fallback
    if not s[0].isalpha():
        s = "o_" + s
    base, i = s, 2
    while s in used:
        s = f"{base[:13]}_{i}"
        i += 1
    used.add(s)
    return s


def _spec_vector(spec):
    types = {o["type"] for o in spec["objects"]}
    v = [1.0 if t in types else 0.0 for t in OBJECTS]
    v += [spec["look"][k] / 100.0 for k in LOOK_KNOBS]
    v += [1.0 if spec["camera"]["style"] == s else 0.0 for s in CAMERA_STYLES]
    return v


def nearest_preset(spec):
    """Nearest Tier 1 preset by distance in normalised spec space."""
    vec = _spec_vector(spec)

    def dist(name):
        pv = _spec_vector(normalise(PRESETS[name], log=None, _allow_snap=False))
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, pv)))
    return min(PRESETS, key=dist)


def normalise(raw, log=print, _allow_snap=True):
    """Any dict (or JSON string) -> resolved spec guaranteed to render.
    Returns (spec) and logs every change it made."""
    notes = []

    def note(msg):
        notes.append(msg)

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            note(f"spec is not valid JSON ({e}); using preset ripple_field")
            raw = {}
    raw = raw if isinstance(raw, dict) else {}
    if isinstance(raw.get("preset"), str) and raw["preset"] in PRESETS:
        base = copy.deepcopy(PRESETS[raw["preset"]])
        base.update({k: v for k, v in raw.items() if k != "preset"})
        raw = base

    # objects
    objects, used_ids, counts = [], set(), {}
    for i, o in enumerate(raw.get("objects") or []):
        if not isinstance(o, dict):
            continue
        t = _snap_enum(o.get("type"), list(OBJECTS), None)
        if t is None:
            note(f"dropped object with unknown type {o.get('type')!r}")
            continue
        if t != o.get("type"):
            note(f"object type {o.get('type')!r} -> {t!r}")
        if OBJECTS[t].get("singleton") and counts.get(t):
            note(f"dropped extra {t} (only one allowed)")
            continue
        counts[t] = counts.get(t, 0) + 1
        objects.append({
            "id": _clean_id(o.get("id"), used_ids, t[:12]),
            "type": t,
            "size": _clamp_int(o.get("size")),
            "density": _snap_enum(o.get("density"), DENSITIES, "medium"),
            "placement": _snap_enum(o.get("placement"), PLACEMENTS, OBJECTS[t]["placement"]),
            "color": _snap_enum(o.get("color"), COLOR_ROLES, "palette"),
        })
    if len(objects) > MAX_OBJECTS:
        note(f"kept the first {MAX_OBJECTS} of {len(objects)} objects")
        objects = objects[:MAX_OBJECTS]
    # map raw ids -> cleaned ids for references
    ids = {o["id"] for o in objects}
    id_of = {o["id"]: o for o in objects}

    def ref(value):
        if value in ids:
            return value
        s = re.sub(r"[^a-z0-9_]", "_", str(value or "").lower()).strip("_")
        if s in ids:
            return s
        # match by type name ("hero_sphere") or prefix
        for o in objects:
            if s and (o["type"] == s or o["id"].startswith(s) or s.startswith(o["id"])):
                return o["id"]
        return None

    # rules: ground/axis objects can't float above or beside the scene
    for o in objects:
        if o["type"] in GROUNDED and o["placement"] not in ("center", "below"):
            note(f"{o['id']} ({o['type']}) placement {o['placement']} -> center (it's a ground-level object)")
            o["placement"] = "center"

    # rules: too many main subjects reads as clutter; keep the camera focus first
    cam_focus = ref((raw.get("camera") or {}).get("focus")) if isinstance(raw.get("camera"), dict) else None
    mains_all = [o for o in objects if OBJECTS[o["type"]]["group"] in ("main", "orbit")]
    if len(mains_all) > MAX_MAIN_SUBJECTS:
        keep = sorted(mains_all, key=lambda o: o["id"] != cam_focus)[:MAX_MAIN_SUBJECTS]
        for o in mains_all:
            if o not in keep:
                objects.remove(o)
                ids.discard(o["id"])
                note(f"dropped {o['id']} ({o['type']}): more than {MAX_MAIN_SUBJECTS} main subjects")

    # rules: fpv needs a flight path; two flight-path objects conflict
    flight = [o for o in objects if o["type"] in ("ring_tunnel", "tower_rows")]
    if len(flight) > 1:
        drop = flight[1]
        objects.remove(drop)
        ids.discard(drop["id"])
        note(f"dropped {drop['id']} ({drop['type']}): only one of ring_tunnel / tower_rows per scene")

    if _allow_snap and not [o for o in objects if OBJECTS[o["type"]]["group"] in ("main", "orbit")]:
        look = raw.get("look") if isinstance(raw.get("look"), dict) else {}
        temp = normalise({"objects": objects or [{"type": "dust"}], "look": look, "camera": raw.get("camera")},
                         log=None, _allow_snap=False)
        name = nearest_preset(temp)
        note(f"no main subject object; snapped objects/camera to nearest preset {name!r} (look kept)")
        snapped = copy.deepcopy(PRESETS[name])
        if look:
            snapped["look"] = {**snapped["look"], **look}
        return normalise(snapped, log=log, _allow_snap=False) if log is None else \
            _finish(normalise(snapped, log=None, _allow_snap=False), notes, log)

    # budget guard: drop density on the heaviest objects first
    def cost(o, mirror_k=1):
        return OBJECTS[o["type"]]["cost"][o["density"]] * mirror_k

    # look
    look_raw = raw.get("look") if isinstance(raw.get("look"), dict) else {}
    look = {"palette": _snap_enum(look_raw.get("palette"), PALETTES, "neon_night")}
    for k, d in (("glow", 70), ("exposure", 60), ("light_rays", 50), ("depth_of_field", 50)):
        look[k] = _clamp_int(look_raw.get(k, d), d)

    # bindings
    bindings = []
    for b in (raw.get("bindings") or [])[:MAX_BINDINGS]:
        if not isinstance(b, dict):
            continue
        obj_id, channel = b.get("object"), b.get("channel")
        if not obj_id and isinstance(b.get("target"), str) and "." in b["target"]:
            obj_id, channel = b["target"].split(".", 1)
        oid = ref(obj_id)
        if oid is None:
            note(f"dropped binding to unknown object {obj_id!r}")
            continue
        chans = list(OBJECTS[id_of[oid]["type"]]["channels"])
        ch = _snap_enum(channel, chans, None)
        if ch is None:
            note(f"dropped binding {oid}.{channel}: {id_of[oid]['type']} has channels {chans}")
            continue
        src_raw = re.sub(r"[^a-z0-9_]", "_", str(b.get("source") or "").lower()).strip("_")
        source = SOURCE_ALIASES.get(src_raw) or _snap_enum(src_raw, list(SOURCES), "energy")
        bindings.append({"source": source, "object": oid,
                         "channel": ch, "amount": _clamp_int(b.get("amount"), 60),
                         "curve": _snap_enum(b.get("curve"), list(CURVES), "linear"),
                         "smoothing": _snap_enum(b.get("smoothing"), list(SMOOTHING), "medium")})

    # relationships
    relationships, positional_subjects, edges = [], set(), {}
    for rel in (raw.get("relationships") or [])[:MAX_RELATIONSHIPS]:
        if not isinstance(rel, dict):
            continue
        verb = _snap_enum(rel.get("verb"), list(VERBS), None)
        if verb is None:
            note(f"dropped relationship with unknown verb {rel.get('verb')!r}")
            continue
        subjects = rel.get("subjects") or ([rel["subject"]] if rel.get("subject") else [])
        subjects = [s for s in (ref(x) for x in subjects) if s]
        subjects = list(dict.fromkeys(subjects))
        anchor = ref(rel.get("anchor"))
        entry = {"verb": verb, "subjects": subjects, "anchor": anchor,
                 "radius": _clamp_int(rel.get("radius"), 50), "speed": _clamp_int(rel.get("speed"), 40),
                 "lag": _clamp_int(rel.get("lag"), 30),
                 "symmetry": _snap_enum(rel.get("symmetry"), list(SYMMETRIES), "radial_3")}
        if verb in POSITIONAL:
            if anchor is None or not subjects:
                note(f"dropped {verb}: needs an existing anchor and subject")
                continue
            subjects = [s for s in subjects if s != anchor and s not in positional_subjects]
            ok = []
            for s in subjects:
                # cycle check: walking anchors from `anchor` must never reach `s`
                cur, seen = anchor, set()
                while cur in edges and cur not in seen:
                    seen.add(cur)
                    cur = edges[cur]
                if cur == s or s in seen:
                    note(f"dropped {verb} {s}->{anchor}: would create a cycle")
                    continue
                edges[s] = anchor
                positional_subjects.add(s)
                ok.append(s)
            if not ok:
                continue
            entry["subjects"] = ok
        elif verb in ("chain", "pulse_together") and len(subjects) < 2:
            note(f"dropped {verb}: needs at least two subjects")
            continue
        elif verb == "mirror" and not subjects:
            note("dropped mirror: no subject")
            continue
        relationships.append(entry)

    mirror_k = {s: SYMMETRIES[r["symmetry"]] for r in relationships if r["verb"] == "mirror" for s in r["subjects"]}
    total = sum(cost(o, mirror_k.get(o["id"], 1)) for o in objects)
    while total > INSTANCE_BUDGET:
        heavy = max((o for o in objects if o["density"] != "low"), key=lambda o: cost(o, mirror_k.get(o["id"], 1)),
                    default=None)
        if heavy is None:
            break
        new = DENSITIES[DENSITIES.index(heavy["density"]) - 1]
        note(f"instance budget: {heavy['id']} density {heavy['density']} -> {new}")
        heavy["density"] = new
        total = sum(cost(o, mirror_k.get(o["id"], 1)) for o in objects)

    # camera
    cam_raw = raw.get("camera") if isinstance(raw.get("camera"), dict) else {}
    style = _snap_enum(cam_raw.get("style"), list(CAMERA_STYLES), "cinematic")
    if style == "fpv" and not [o for o in objects if o["type"] in ("ring_tunnel", "tower_rows")]:
        note("camera fpv needs ring_tunnel or tower_rows; using cinematic")
        style = "cinematic"
    mains = [o["id"] for o in objects if OBJECTS[o["type"]]["group"] == "main"] or [objects[0]["id"]]
    focus = ref(cam_raw.get("focus")) or mains[0]
    camera = {"style": style, "cuts": _snap_enum(cam_raw.get("cuts"), list(CUTS), "on_sections"),
              "shake": _clamp_int(cam_raw.get("shake"), 30), "focus": focus}

    # combination rules on look
    if look["glow"] > 85 and any(o["type"] == "wire_cage" for o in objects) and look["palette"] in ("acid", "ember"):
        note("glow capped at 80: wire_cage + very high glow on a hot palette blows out")
        look["glow"] = 80

    spec = {"spec": SPEC_VERSION, "objects": objects, "bindings": bindings, "relationships": relationships,
            "camera": camera, "look": look}
    return _finish(spec, notes, log)


def _finish(spec, notes, log):
    if log:
        for n in notes:
            log(f"spec: {n}")
    return spec
