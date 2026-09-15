"""Compile a resolved spec (vocab.normalise output) into a renderable scene.

Pipeline, all precomputed once over every frame before rendering:
  1. audio sources -> binding arrays: smoothing, response curve, amount; bindings
     summed per object channel and clamped; unbound channels get the object
     type's default binding so nothing is ever dead
  2. relationships: chain (staggered delay), pulse_together (shared envelope),
     placement + orbit / follow (per-frame origins, anchors resolved before
     subjects), mirror (k copies rotated about the world centre)
  3. camera: shot list from the style around the focus object; cuts from the
     cuts enum
update(f) then only indexes arrays and fills instance buffers.
"""
import math

import numpy as np

from objects import COMPONENTS, lerp
from vocab import OBJECTS, SYMMETRIES

SMOOTHING_S = {"tight": (0.005, 0.06), "medium": (0.03, 0.25), "loose": (0.15, 0.8)}


def _smooth(x, fps, attack_s, release_s):
    k_att = 1.0 if attack_s <= 0 else 1 - math.exp(-1.0 / (fps * attack_s))
    k_rel = 1 - math.exp(-1.0 / (fps * release_s))
    out = np.empty_like(x)
    s = 0.0
    for i, v in enumerate(x):
        s += (v - s) * (k_att if v > s else k_rel)
        out[i] = s
    return out


def _curve(x, name):
    x = np.clip(x, 0.0, 1.0)
    if name == "s_curve":
        return x * x * (3 - 2 * x)
    if name == "punchy":
        return x ** 2.2
    if name == "smooth":
        return np.sin(x * math.pi / 2)
    return x


def source_arrays(sig):
    beat = np.asarray(sig.beat)
    bi = np.asarray(sig.beat_index)
    return {
        "bass": np.asarray(sig.low), "mid": np.asarray(sig.mid), "treble": np.asarray(sig.high),
        "energy": np.asarray(sig.energy), "onset": np.asarray(sig.onset), "beat": beat,
        "downbeat": np.where((bi >= 0) & (bi % 4 == 0), beat, 0.0),
        "brightness": np.asarray(sig.brightness),
        "section_change": np.exp(-np.asarray(sig.section_elapsed) / 1.5),
    }


class Shot:
    def __init__(self, lens, pos, target, shift=(0.0, 0.0), roll=0.0):
        self.lens, self.pos, self.target, self.shift, self.roll = lens, pos, target, shift, roll


class SpecScene:
    def __init__(self, spec, log=print):
        self.spec, self.log = spec, log

    # ── build ────────────────────────────────────────────────────────────────
    def build(self, r, sig, k):
        self.r, self.sig, self.k = r, sig, k
        spec, n, fps = self.spec, sig.n, sig.fps
        src = source_arrays(sig)

        # 1. channels
        channels = {o["id"]: {} for o in spec["objects"]}
        bound = {}
        for b in spec["bindings"]:
            key = (b["object"], b["channel"])
            att, rel = SMOOTHING_S[b["smoothing"]]
            contrib = (b["amount"] / 100.0) * _curve(_smooth(src[b["source"]], fps, att, rel), b["curve"])
            bound[key] = bound.get(key, 0.0) + contrib
        for o in spec["objects"]:
            for ch, (s_name, amount, curve, smoothing) in OBJECTS[o["type"]]["defaults"].items():
                arr = bound.get((o["id"], ch))
                if arr is None:
                    att, rel = SMOOTHING_S[smoothing]
                    arr = (amount / 100.0) * _curve(_smooth(src[s_name], fps, att, rel), curve)
                channels[o["id"]][ch] = np.clip(arr, 0.0, 1.0)

        for rel in spec["relationships"]:
            subs = [s for s in rel["subjects"] if s in channels]
            if rel["verb"] == "chain":
                lag = int(round(lerp(0.0, 0.6, rel["lag"] / 100.0) * fps))
                for i, s in enumerate(subs):
                    shift = i * lag
                    if shift:
                        for ch, arr in channels[s].items():
                            channels[s][ch] = np.concatenate([np.full(shift, arr[0]), arr[:-shift]])
            elif rel["verb"] == "pulse_together" and len(subs) > 1:
                prim = {s: next(iter(OBJECTS[self._type(s)]["channels"])) for s in subs}
                shared = np.max(np.stack([channels[s][prim[s]] for s in subs]), axis=0)
                for s in subs:
                    channels[s][prim[s]] = shared

        # 2. components (sizes first, so layout can use scene extent)
        mains = [o for o in spec["objects"] if OBJECTS[o["type"]]["group"] in ("main", "orbit")]
        probe_extent = 6.0
        self.components = {}
        order = sorted(spec["objects"], key=lambda o: OBJECTS[o["type"]]["group"] in ("frame", "backdrop"))
        main_extent = None
        for o in order:
            if OBJECTS[o["type"]]["group"] in ("frame", "backdrop") and main_extent is None:
                main_extent = max([c.extent for c in self.components.values()] or [probe_extent])
            o = dict(o)
            if o["type"] == "halo_ring":
                compact = any(m["type"] in ("hero_sphere", "wire_cage", "shard_swarm", "double_helix") for m in mains)
                o["_tilt"] = (75.0, 0.0, 20.0) if compact else (0.0, 0.0, 0.0)
            comp = COMPONENTS[o["type"]](r, sig, o, channels[o["id"]], k, main_extent or probe_extent)
            comp.build()
            self.components[o["id"]] = comp
        self.main_extent = main_extent or max([c.extent for c in self.components.values()] or [probe_extent])
        E = self.main_extent
        floor_z = min([self.components[m["id"]].floor_z for m in mains] or [-1.0])

        # placement
        place = {"center": (0, 0, 0), "above": (0, 0, 1.3 * E + 2), "below": (0, 0, -0.8 * E - 1),
                 "left": (-1.9 * E - 2, 0, 0), "right": (1.9 * E + 2, 0, 0), "background": (0, 3.0 * E + 150, 20),
                 "floor": (0, 0, 0)}
        for o in spec["objects"]:
            comp = self.components[o["id"]]
            origin = np.array(place[o["placement"]], dtype=np.float64)
            if o["type"] == "sun_disc" and o["placement"] != "background":
                origin = origin + np.array([0, 3.0 * E + 150, 20])
            comp.origins[:] = origin
            if o["type"] == "grid_floor":
                comp.z = floor_z
                comp.step = 4.0 if any(m["type"] == "tower_rows" for m in mains) else 2.0

        # orbit / follow, anchors first
        pos_rels = {s: rel for rel in spec["relationships"] if rel["verb"] in ("orbit", "follow") for s in rel["subjects"]}
        done = set()

        def resolve(sid, depth=0):
            if sid in done or depth > 8:
                return
            rel = pos_rels.get(sid)
            if rel:
                resolve(rel["anchor"], depth + 1)
                anchor, comp = self.components[rel["anchor"]], self.components[sid]
                if rel["verb"] == "orbit":
                    radius = lerp(1.1, 2.6, rel["radius"] / 100.0) * anchor.extent + 0.3 * comp.extent
                    rate = lerp(0.05, 0.9, rel["speed"] / 100.0) * (0.6 + 0.8 * src["energy"])
                    angle = np.concatenate([[0.0], np.cumsum(rate[:-1])]) / fps + (hash(sid) % 628) / 100.0
                    comp.origins[:] = anchor.origins + np.stack(
                        [radius * np.cos(angle), radius * np.sin(angle), np.zeros(n)], 1)
                else:
                    lag = int(round(lerp(0.1, 1.5, rel["lag"] / 100.0) * fps))
                    idx = np.clip(np.arange(n) - lag, 0, n - 1)
                    comp.origins[:] = anchor.origins[idx] + np.array([0.0, -1.2 * anchor.extent, 0.0])
            done.add(sid)

        for sid in list(self.components):
            resolve(sid)

        # mirror: extra copies rotated about the world centre
        self.extra = []
        for rel in spec["relationships"]:
            if rel["verb"] != "mirror":
                continue
            copies = SYMMETRIES[rel["symmetry"]]
            for sid in rel["subjects"]:
                src_comp = self.components[sid]
                o = next(x for x in spec["objects"] if x["id"] == sid)
                for i in range(1, copies):
                    comp = COMPONENTS[o["type"]](r, sig, dict(o), channels[sid], k, self.main_extent)
                    comp.build()
                    if rel["symmetry"] == "x":
                        comp.flip = True
                        comp.origins[:] = src_comp.origins * np.array([-1.0, 1.0, 1.0])
                    else:
                        a = 2 * math.pi * i / copies
                        c, s = math.cos(a), math.sin(a)
                        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                        comp.origins[:] = src_comp.origins @ rot.T
                        comp.yaw = a
                    self.extra.append(comp)

        # 3. camera
        self.focus = self.components[spec["camera"]["focus"]]
        self.shots = self._shots()
        self.cut, self.since_cut = self._cuts()
        self.shake = spec["camera"]["shake"] / 50.0
        self.log(f"spec scene: {len(self.components)} objects (+{len(self.extra)} mirror copies), "
                 f"main extent {E:.1f}, camera {spec['camera']['style']} with {len(self.shots)} shots, "
                 f"cuts {spec['camera']['cuts']}")

    def _type(self, sid):
        return next(o["type"] for o in self.spec["objects"] if o["id"] == sid)

    def _focus_point(self, f):
        return self.focus.origin_at(f) + self.focus.focus_offset(f)

    def _shots(self):
        E, style = self.main_extent, self.spec["camera"]["style"]
        fp = self._focus_point
        floor = min([c.floor_z for c in self.components.values() if c.extent] or [-1.0])

        def orbit(radius, dz, angle0, speed):
            return lambda f, u: fp(f) + np.array([radius * math.cos(angle0 + speed * u),
                                                  radius * math.sin(angle0 + speed * u), dz])

        def offset(dx, dy, dz, drift=(0.0, 0.0, 0.0)):
            return lambda f, u: fp(f) + np.array([dx + drift[0] * u, dy + drift[1] * u, dz + drift[2] * u])

        target = lambda f, u: fp(f)  # noqa: E731

        def orbit_wide(i):
            return Shot(24, orbit(1.5 * E + 5, 0.35 * E + 1.5, math.radians(-50 + 120 * i), 0.04), target,
                        shift=(0.12 if i % 2 == 0 else -0.12, 0.0))

        def low_dolly(i):
            side = 1 if i % 2 == 0 else -1
            # Low, but high enough to see over the nearest geometry: eye at a
            # quarter of the scene height above the floor, aimed slightly up.
            ground = floor + max(1.5, 0.4 * E) - fp(0)[2]
            return Shot(22, offset(0.3 * side * E, -1.2 * E - 4, ground, drift=(0, 0.35, 0)),
                        offset(0, 0.5 * E, floor - fp(0)[2] + 0.05 * E, drift=(0, 0.35, 0)),
                        roll=6 * side, shift=(0.0, 0.12))

        def crane(i):
            return Shot(85, orbit(3.5 * E + 25, 2.0 * E + 18, math.radians(20 + 110 * i), 0.02), target)

        def tele(i):
            return Shot(135, orbit(5.0 * E + 35, 0.2 * E + 3, math.radians(30 + 140 * i), 0.01), target,
                        shift=(0.1 * (1 if i % 2 else -1), 0.0))

        def top_down(i):
            return Shot(50, offset(0.2 * E + 1, -0.4 * E - 2, 1.6 * E + 14), target, roll=28 + 40 * i,
                        shift=(0.0, -0.1))

        tunnel = next((c for c in self.components.values() if type(c).__name__ == "RingTunnel"), None)
        street = next((c for c in self.components.values() if type(c).__name__ == "TowerRows"), None)
        if style == "fpv" and tunnel is not None:
            o = tunnel.origin_at
            tr = tunnel.travel
            return [Shot(18, lambda f, u: o(f) + [0.0, tr[f], 0.0], lambda f, u: o(f) + [0.0, tr[f] + 15, 0.0]),
                    Shot(28, lambda f, u: o(f) + [2.2, tr[f], -1.8], lambda f, u: o(f) + [0.4, tr[f] + 20, 0.4], roll=14),
                    Shot(12, lambda f, u: o(f) + [0.0, tr[f], 1.6], lambda f, u: o(f) + [0.0, tr[f] + 10, 1.6], roll=-20)]
        if style == "fpv" and street is not None:
            o = street.origin_at
            return [Shot(24, lambda f, u: o(f) + [0.0, -8.0 + 0.5 * u, 1.1], lambda f, u: o(f) + [0, 40 + 0.5 * u, 3.5],
                         shift=(0.0, 0.08)),
                    Shot(45, lambda f, u: o(f) + [10.0, 6.0 + 0.3 * u, 2.5], lambda f, u: o(f) + [-1, 30, 3],
                         shift=(0.12, 0.0)),
                    Shot(35, lambda f, u: o(f) + [0.0, -12.0, 14.0], lambda f, u: o(f) + [0, 35, 0]),
                    Shot(85, lambda f, u: o(f) + [-2.0, -40.0, 3.0], lambda f, u: o(f) + [0, 60, 6])]
        makers = {"orbit_wide": orbit_wide, "low_dolly": low_dolly, "crane": crane, "tele": tele,
                  "top_down": top_down}
        if style in makers:
            return [makers[style](i) for i in range(3)]
        return [orbit_wide(0), tele(0), low_dolly(0), top_down(0), orbit_wide(1), crane(1)]

    def _cuts(self):
        sig, mode = self.sig, self.spec["camera"]["cuts"]
        if mode == "on_sections":
            return np.asarray(sig.cut), np.asarray(sig.since_cut)
        n = sig.n
        if mode == "never":
            return np.zeros(n, dtype=int), np.asarray(sig.t) - sig.t[0]
        bi = np.maximum(np.asarray(sig.beat_index), 0)
        cut = bi // 32
        since = np.zeros(n)
        start = 0
        for f in range(1, n):
            if cut[f] != cut[f - 1]:
                start = f
            since[f] = (f - start) / sig.fps
        return cut, since

    # ── per frame ────────────────────────────────────────────────────────────
    def update(self, f):
        self.r.floor = None
        for comp in list(self.components.values()) + self.extra:
            comp.update(f)
        self.r.hero = tuple(float(v) for v in self._focus_point(f))
        suns = [c for c in self.components.values() if type(c).__name__ == "SunDisc"]
        if suns:
            self.r.hero = tuple(float(v) for v in suns[0].origin_at(f))

        sig = self.sig
        shot = self.shots[int(self.cut[f]) % len(self.shots)]
        u = float(self.since_cut[f])
        eye = np.asarray(shot.pos(f, u), dtype=np.float64)
        target = np.asarray(shot.target(f, u), dtype=np.float64)
        amp = 0.06 * self.shake * (0.5 * sig.low[f] * sig.beat[f] + 0.5 * sig.onset[f]) * max(1.0, self.main_extent / 6)
        t = sig.t[f]
        eye = eye + np.array([math.sin(t * 37.0), math.sin(t * 29.0 + 1.3), math.sin(t * 41.0 + 2.1)]) * amp
        far = max(500.0, 12.0 * self.main_extent + 300.0)
        if any(type(c).__name__ == "RingTunnel" for c in self.components.values()):
            far = max(far, max(c.length for c in self.components.values() if hasattr(c, "length")) + 40)
        self.r.clip_far = far
        self.r.camera = {"eye": eye, "target": target, "lens": shot.lens, "shift": shot.shift, "roll": shot.roll}
