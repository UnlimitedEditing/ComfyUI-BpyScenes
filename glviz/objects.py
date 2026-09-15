"""Object components for spec-built visualizers (the "nouns").

Each component owns its batches and turns per-frame channel values (0-1 arrays
compiled from audio bindings, see spec_scene.py) into instance data. Every
channel maps onto a range tuned to look acceptable at both ends, and `size` /
`density` map to tuned layouts, so no combination of spec values can produce
degenerate geometry. Components never set colours: they write drive/accent.

Placement: each component draws around self.origin_at(f) (a per-frame array so
relationships like orbit/follow can move it) rotated by self.yaw and optionally
mirrored in x (for the mirror verb).
"""
import math

import numpy as np

from gpu import box_mesh, disc_mesh, hash01, ico_mesh, ring_mesh, wire_mesh


def lerp(a, b, t):
    return a + (b - a) * t


def _tilt(rx, ry, rz):
    from gpu import model_matrices
    return model_matrices([[0, 0, 0]], rot=[[math.radians(rx), math.radians(ry), math.radians(rz)]])[0] \
        .reshape(4, 4).T[:3, :3]


class Component:
    def __init__(self, r, sig, spec, channels, k, scene_extent=6.0):
        self.r, self.sig, self.spec, self.ch, self.k = r, sig, spec, channels, k
        self.size = spec["size"] / 100.0
        self.density = spec["density"]
        self.accent_role = 1.0 if spec["color"] == "accent" else 0.0
        self.scene_extent = scene_extent
        self.origins = np.zeros((sig.n, 3))
        self.yaw, self.flip = 0.0, False
        self.batches = []

    # layout helpers
    def batch(self, mesh, count):
        b = self.r.batch(mesh, count)
        self.batches.append(b)
        return b

    def origin_at(self, f):
        return self.origins[f]

    def place(self, local, f):
        """local (n,3) points around the object's own origin -> world."""
        p = np.array(local, dtype=np.float64, copy=True)
        if self.flip:
            p[:, 0] = -p[:, 0]
        if self.yaw:
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            p[:, 0], p[:, 1] = c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]
        return p + self.origin_at(f)

    def c(self, name, f):
        return float(self.ch[name][f])

    def cumulative(self, rate):
        """Integrate a per-frame rate array (units/s)."""
        return np.concatenate([[0.0], np.cumsum(rate[:-1])]) / self.sig.fps

    # overridden
    extent = 5.0
    floor_z = -1.0

    def focus_offset(self, f):
        return np.zeros(3)

    def build(self):
        pass

    def update(self, f):
        pass


class ParticleField(Component):
    N = {"low": 12, "medium": 18, "high": 26}

    def build(self):
        n = self.N[self.density]
        self.sp = lerp(0.4, 0.75, self.size)
        self.scale_u = self.sp / 0.55
        half = (n - 1) / 2
        gx, gy = np.meshgrid((np.arange(n) - half) * self.sp, (np.arange(n) - half) * self.sp, indexing="ij")
        self.xy = np.stack([gx.ravel(), gy.ravel()], 1)
        self.dist = np.hypot(self.xy[:, 0], self.xy[:, 1])
        self.idx = np.arange(n * n)
        self.b = self.batch(ico_mesh(0.16 * self.scale_u, 1)[0], n * n)
        self.phase = self.cumulative(0.25 + 1.0 * self.ch["wave_speed"] * self.k)
        self.extent = n * self.sp / 2 * 1.1
        self.floor_z = -2.2 * self.scale_u

    def update(self, f):
        sig, k, u = self.sig, self.k, self.scale_u
        amp = (0.1 + 1.0 * self.c("height", f) * k) * u
        pulse_h = 0.9 * k * self.c("pulse", f) * u
        ring = np.exp(-((self.dist - sig.since_beat[f] * 7.0 * u) / (0.7 * u)) ** 2)
        z = amp * np.sin(2 * np.pi * (self.phase[f] - self.dist / (6.0 * u))) + pulse_h * ring
        drive = 0.5 + 0.5 * z / (amp + pulse_h + 1e-6)
        drive = drive + np.where(hash01(self.idx, sig.beat_index[f], 11) > 0.9, 0.5 * self.c("sparkle", f), 0.0)
        local = np.column_stack([self.xy, z])
        self.b.pos[:] = self.place(local, f)
        self.b.scale[:] = (0.6 + 0.6 * np.clip(drive, 0, 1))[:, None]
        self.b.drive = drive
        self.b.accent = np.maximum(self.accent_role, ring * self.c("pulse", f) * 1.5)


class PillarGrid(Component):
    N = {"low": 8, "medium": 12, "high": 16}
    FIRE_P = 0.2

    def build(self):
        n = self.N[self.density]
        self.sp = lerp(1.0, 1.6, self.size)
        half = (n - 1) / 2
        gx, gy = np.meshgrid((np.arange(n) - half) * self.sp, (np.arange(n) - half) * self.sp, indexing="ij")
        self.xy = np.stack([gx.ravel(), gy.ravel()], 1)
        self.idx = np.arange(n * n)
        self.b = self.batch(box_mesh(0.69 * self.sp, 0.69 * self.sp, 1.0), n * n)
        self.w = 0.6 + 1.2 * hash01(self.idx, 1)
        self.ph = 6.28 * hash01(self.idx, 2)
        self.hscale = self.sp / 1.3
        self.extent = n * self.sp / 2
        self.floor_z = 0.0

    def focus_offset(self, f):
        return np.array([0.0, 0.0, 1.2 * self.hscale])

    def update(self, f):
        sig, k = self.sig, self.k
        bi, t = sig.beat_index[f], sig.t[f]
        h = 0.15 + 1.2 * k * self.c("height", f) * (0.25 + 0.75 * (0.5 + 0.5 * np.sin(t * self.w + self.ph)))
        fired = (hash01(bi, self.idx, 3) < self.FIRE_P) if bi >= 0 else np.zeros(len(self.idx), bool)
        h = h + np.where(fired, 3.5 * k * self.c("fire", f), 0.0)
        self.b.pos[:] = self.place(np.column_stack([self.xy, np.zeros(len(self.idx))]), f)
        self.b.scale[:, 0] = self.b.scale[:, 1] = 1.0
        self.b.scale[:, 2] = np.maximum(h * self.hscale, 0.02)
        self.b.drive = h / (0.15 + 4.7 * k) + 0.3 * self.c("sparkle", f) * hash01(self.idx, bi, 9)
        self.b.accent = np.maximum(self.accent_role, np.where(fired & (bi >= 0 and bi % 4 == 0), 1.0, 0.0))


class RingTunnel(Component):
    RINGS = {"low": 24, "medium": 40, "high": 60}
    STREAKS = {"low": 36, "medium": 64, "high": 100}
    SPACING = 3.0

    def build(self):
        n = self.RINGS[self.density]
        self.radius = lerp(3.5, 6.0, self.size)
        self.length = n * self.SPACING
        self.travel = self.cumulative((3.0 + 10.0 * self.ch["speed"]) * (0.5 + 0.5 * self.k))
        self.twist = self.cumulative(0.15 + 1.3 * self.ch["twist"] * self.k)
        self.i = np.arange(n)
        self.rings = self.batch(ring_mesh(self.radius, 0.12, 6, 4), n)
        ns = self.STREAKS[self.density]
        s = np.arange(ns)
        self.s_angle = 2 * np.pi * hash01(s, 21)
        self.s_radius = self.radius * (0.75 + 0.2 * hash01(s, 22))
        self.s_offset = self.length * hash01(s, 23)
        self.s_idx = s
        self.streaks = self.batch(box_mesh(0.05, 1.8, 0.05, base=False), ns)
        self.exit = self.batch(disc_mesh(2.5), 1)
        self.extent = self.radius
        self.floor_z = -self.radius - 1.0

    def focus_offset(self, f):
        return np.array([0.0, self.travel[f] + 15.0, 0.0])

    def update(self, f):
        sig, k = self.sig, self.k
        cam_y = self.travel[f]
        rel = (self.i * self.SPACING - cam_y) % self.length - 2.0
        bump = np.exp(-((rel - sig.since_beat[f] * 28.0) / 3.0) ** 2) * self.c("pulse", f)
        s = 1.0 + 0.2 * self.c("speed", f) * k + 0.3 * bump * k
        n = len(self.i)
        self.rings.pos[:] = self.place(np.column_stack([np.zeros(n), cam_y + rel, np.zeros(n)]), f)
        self.rings.rot = np.column_stack([np.full(n, math.pi / 2), self.twist[f] + self.i * 0.09,
                                          np.full(n, self.yaw)])
        self.rings.scale[:] = s[:, None]
        self.rings.drive = 0.1 + 0.9 * bump + 0.25 * self.c("sparkle", f) * hash01(self.i, sig.beat_index[f], 4)
        self.rings.accent = np.maximum(self.accent_role, np.where(self.i % 4 == 0, 1.0, 0.0))
        srel = (self.s_offset - cam_y * 1.6) % self.length
        local = np.column_stack([self.s_radius * np.cos(self.s_angle), cam_y + srel,
                                 self.s_radius * np.sin(self.s_angle)])
        self.streaks.pos[:] = self.place(local, f)
        self.streaks.drive = self.c("sparkle", f) * np.where(hash01(self.s_idx, sig.beat_index[f], 5) > 0.5, 1.0, 0.2)
        self.exit.pos[:] = self.place([[0.0, cam_y + self.length - 4.0, 0.0]], f)
        self.exit.drive[:] = 0.5 + 0.5 * self.c("speed", f)


class HeroSphere(Component):
    def build(self):
        self.radius = lerp(0.8, 2.2, self.size)
        self.b = self.batch(ico_mesh(self.radius, 3)[0], 1)
        self.spin = self.cumulative(0.1 + 0.6 * self.ch["spin"] * self.k)
        self.extent = self.radius * 1.4
        self.floor_z = -self.radius * 2.0

    def update(self, f):
        s = 1.0 + 0.35 * self.k * self.c("scale", f)
        self.b.pos[:] = self.place([[0, 0, 0]], f)
        self.b.scale[:] = s
        self.b.rot = np.array([[0.3 * self.spin[f], 0.0, self.spin[f] + self.yaw]])
        self.b.drive[:] = 0.12 + 0.6 * self.c("glow", f)
        self.b.accent[:] = self.accent_role


class WireCage(Component):
    def build(self):
        self.radius = lerp(1.2, 3.2, self.size)
        _, v, faces = ico_mesh(self.radius, 1)
        self.b = self.batch(wire_mesh(v, faces, 0.02 + 0.015 * self.radius), 1)
        self.spin = self.cumulative(0.1 + 0.6 * self.ch["spin"] * self.k)
        self.extent = self.radius * 1.2
        self.floor_z = -self.radius * 1.6

    def update(self, f):
        s = 1.0 + 0.15 * self.k * self.c("scale", f)
        self.b.pos[:] = self.place([[0, 0, 0]], f)
        self.b.scale[:] = s
        self.b.rot = np.array([[0.0, -0.4 * self.spin[f], -self.spin[f] + self.yaw]])
        self.b.drive[:] = 0.15 + 0.7 * self.c("glow", f)
        self.b.accent[:] = self.accent_role


class ShardSwarm(Component):
    ORBITS = [(3.8, 0.25, (20, 0, 0)), (5.5, 0.33, (-35, 0, 40)), (7.5, 0.42, (60, 0, -30))]
    TOTAL = {"low": 60, "medium": 120, "high": 220}

    def build(self):
        total = self.TOTAL[self.density]
        rs = lerp(0.6, 1.4, self.size)
        radii, a0, speed, oid, jid = [], [], [], [], []
        for o, (radius, frac, _) in enumerate(self.ORBITS):
            count = int(total * frac) if o < 2 else total - sum(int(total * fr) for _, fr, _ in self.ORBITS[:2])
            j = np.arange(count)
            radii.append(radius * rs * (0.92 + 0.16 * hash01(o, j, 6)))
            a0.append(2 * np.pi * j / max(count, 1))
            speed.append(np.full(count, 3.8 / radius))
            oid.append(np.full(count, o))
            jid.append(j)
        self.radii, self.a0, self.speed = map(np.concatenate, (radii, a0, speed))
        self.oid, self.jid = np.concatenate(oid), np.concatenate(jid)
        self.ids = self.oid * 1000 + self.jid
        self.tilts = np.stack([_tilt(*t) for _, _, t in self.ORBITS])[self.oid]
        self.b = self.batch(ico_mesh(0.14 * rs, 0)[0], total)
        self.b.scale[:] = np.stack([0.6 + 1.4 * hash01(self.oid, self.jid, n) for n in (1, 2, 3)], 1)
        self.b.rot = np.stack([6.28 * hash01(self.oid, self.jid, 4), 6.28 * hash01(self.oid, self.jid, 5),
                               np.zeros(total)], 1)
        self.phase = self.cumulative(0.12 + 0.8 * self.ch["speed"] * self.k)
        self.extent = 7.5 * rs * 1.1
        self.floor_z = -self.extent

    def update(self, f):
        sig = self.sig
        a = self.a0 + self.phase[f] * self.speed
        local = np.stack([self.radii * np.cos(a), self.radii * np.sin(a), np.zeros_like(a)], 1)
        local = np.einsum("nij,nj->ni", self.tilts, local)
        self.b.pos[:] = self.place(local, f)
        b = sig.beat_index[f]
        flash = np.where(hash01(self.ids, b, 2) > 0.7, self.c("flash", f), 0.0)
        self.b.drive = 0.08 + 0.6 * self.c("sparkle", f) * hash01(self.ids, b, 3) + 0.8 * flash
        self.b.accent = np.maximum(self.accent_role, np.where(self.oid == 2, 1.0, 0.0))


class TowerRows(Component):
    TOWERS = {"low": 30, "medium": 44, "high": 64}
    SPACING, DELAY_S = 2.2, 0.05

    def build(self):
        n = self.TOWERS[self.density]
        j = np.arange(n)
        self.n = n
        self.x = lerp(2.6, 4.2, self.size)
        self.hmax = lerp(0.6, 1.3, self.size) * 7.0
        self.y = 3.0 + j * self.SPACING
        self.towers = self.batch(box_mesh(1.0, 1.0, 1.0), n * 2)
        self.dashes = self.batch(box_mesh(0.12, 1.0, 0.02), n)
        self.delay = (j * self.DELAY_S * self.sig.fps).astype(int)
        self.left, self.right = self.ch["height_left"], self.ch["height_right"]
        self.dash = self.ch["dash_pulse"]
        self.extent = 12.0
        self.floor_z = -0.01

    def focus_offset(self, f):
        return np.array([0.0, 40.0, 3.5])

    def update(self, f):
        hist = np.clip(f - self.delay, 0, self.sig.n - 1)
        h = 0.2 + self.hmax * self.k * np.concatenate([self.left[hist], self.right[hist]])
        local = np.column_stack([np.concatenate([np.full(self.n, -self.x), np.full(self.n, self.x)]),
                                 np.tile(self.y, 2), np.zeros(2 * self.n)])
        self.towers.pos[:] = self.place(local, f)
        self.towers.scale[:, 2] = h
        self.towers.drive = h / (0.2 + self.hmax * self.k)
        self.towers.accent = np.maximum(self.accent_role, np.concatenate([np.zeros(self.n), np.ones(self.n)]))
        self.dashes.pos[:] = self.place(np.column_stack([np.zeros(self.n), self.y, np.zeros(self.n)]), f)
        self.dashes.drive = 0.05 + 0.9 * self.dash[hist]


class DoubleHelix(Component):
    PER = {"low": 40, "medium": 64, "high": 96}
    RISE = 0.38

    def build(self):
        per = self.PER[self.density]
        self.radius = lerp(1.6, 3.2, self.size)
        self.height = per * self.RISE
        jj = np.tile(np.arange(per), 2)
        strand = np.concatenate([np.zeros(per), np.ones(per)])
        a = jj * 0.35 + strand * math.pi
        self.local = np.stack([self.radius * np.cos(a), self.radius * np.sin(a), jj * self.RISE], 1)
        self.strand, self.jj = strand, jj
        self.spheres = self.batch(ico_mesh(0.22 * self.radius / 2.4, 2)[0], per * 2)
        rj = np.arange(0, per, 4)
        self.rung_z, self.rung_a = rj * self.RISE, rj * 0.35
        self.rungs = self.batch(box_mesh(2 * self.radius, 0.06, 0.06, base=False), len(rj))
        self.halos = self.batch(ring_mesh(self.radius * 2.1, 0.05, 96, 6), 4)
        self.halo_z = np.linspace(0.12, 0.85, 4) * self.height
        self.spin = self.cumulative(0.25 + 1.2 * self.ch["spin"] * self.k)
        self.extent = self.radius * 2.2
        self.floor_z = -0.6

    def focus_offset(self, f):
        return np.array([0.0, 0.0, self.height * 0.5])

    def update(self, f):
        sig, k = self.sig, self.k
        spin = self.spin[f] + self.yaw
        c, s = math.cos(spin), math.sin(spin)
        rot = self.local.copy()
        rot[:, 0], rot[:, 1] = c * self.local[:, 0] - s * self.local[:, 1], s * self.local[:, 0] + c * self.local[:, 1]
        origin = self.origin_at(f)
        self.spheres.pos[:] = rot + origin
        front = (sig.since_beat[f] * 20.0) % (self.height + 6)
        bump = np.exp(-((self.local[:, 2] - front) / 1.2) ** 2) * self.c("pulse", f)
        self.spheres.scale[:] = (1.0 + 1.0 * bump * k + 0.3 * self.c("swell", f) * k)[:, None]
        self.spheres.drive = 0.1 + 0.8 * bump + 0.3 * self.c("sparkle", f) * hash01(self.strand, self.jj,
                                                                                      sig.beat_index[f])
        self.spheres.accent = np.maximum(self.accent_role, self.strand)
        self.rungs.pos[:] = origin + np.column_stack([np.zeros_like(self.rung_z), np.zeros_like(self.rung_z),
                                                      self.rung_z])
        self.rungs.rot = np.column_stack([np.zeros_like(self.rung_a), np.zeros_like(self.rung_a), self.rung_a + spin])
        self.rungs.drive[:] = 0.05 + 0.6 * self.c("spin", f)
        self.rungs.accent[:] = 0.5
        hs = 1.0 + 0.25 * self.c("swell", f) * k * (1.0 - 0.15 * np.arange(4))
        self.halos.pos[:] = origin + np.column_stack([np.zeros(4), np.zeros(4), self.halo_z])
        self.halos.scale[:, 0], self.halos.scale[:, 1] = hs, hs
        self.halos.drive[:] = 0.08 + 0.6 * self.c("swell", f)
        self.halos.accent[:] = 1.0


class HaloRing(Component):
    def build(self):
        self.radius = lerp(0.9, 1.9, self.size) * self.scene_extent
        self.b = self.batch(ring_mesh(self.radius, 0.04 + 0.006 * self.radius, 96, 6), 1)
        self.tilt = self.spec.get("_tilt", (0.0, 0.0, 0.0))
        self.extent = self.radius
        self.floor_z = -1.0

    def update(self, f):
        s = 1.0 + 0.12 * self.c("scale", f) * self.k
        self.b.pos[:] = self.place([[0, 0, -0.15 * self.scene_extent if self.tilt == (0.0, 0.0, 0.0) else 0.0]], f)
        self.b.scale[:] = (s, s, 1.0)
        self.b.rot = np.radians([[self.tilt[0], self.tilt[1], self.tilt[2] + math.degrees(self.yaw)]])
        self.b.drive[:] = 0.12 + 0.75 * self.c("glow", f)
        self.b.accent[:] = self.accent_role


class SunDisc(Component):
    def build(self):
        self.radius = lerp(15.0, 35.0, self.size)
        self.b = self.batch(disc_mesh(self.radius), 1)
        self.extent = self.radius
        self.floor_z = 0.0

    def update(self, f):
        s = 1.0 + 0.1 * self.c("scale", f) * self.k
        self.b.pos[:] = self.place([[0.0, 0.0, 0.0]], f)
        self.b.scale[:] = s
        self.b.drive[:] = 0.55 + 0.45 * self.c("glow", f)
        self.b.accent[:] = self.accent_role


class Dust(Component):
    COUNT = {"low": 80, "medium": 150, "high": 300}

    def build(self):
        n = self.COUNT[self.density]
        i = np.arange(n)
        e = self.scene_extent
        r_min, r_max = 2.0 * e + 10.0, 4.0 * e + 25.0
        a = 2 * np.pi * hash01(3, i, 1)
        rad = r_min + (r_max - r_min) * hash01(3, i, 2)
        self.base = np.stack([rad * np.cos(a), rad * np.sin(a), -0.3 * r_min + 0.9 * r_max * hash01(3, i, 3)], 1)
        self.b = self.batch(ico_mesh(0.07, 0)[0], n)
        s = (0.6 + 1.8 * hash01(3, i, 4)) * lerp(0.7, 1.6, self.size)
        self.b.scale[:] = np.stack([s, s, s], 1)
        self.idx = i
        self.spin = self.cumulative(0.01 + 0.05 * self.ch["drift"])
        self.extent = r_max
        self.floor_z = 0.0

    def update(self, f):
        c, s = math.cos(self.spin[f]), math.sin(self.spin[f])
        p = self.base.copy()
        p[:, 0], p[:, 1] = c * self.base[:, 0] - s * self.base[:, 1], s * self.base[:, 0] + c * self.base[:, 1]
        self.b.pos[:] = p + self.origin_at(f)
        flicker = self.c("sparkle", f) * np.where(hash01(self.idx, self.sig.beat_index[f], 7) > 0.6, 1.0, 0.25)
        self.b.drive = 0.05 + 0.8 * flicker
        self.b.accent[:] = self.accent_role


class GridFloor(Component):
    def build(self):
        self.extent = 0.0
        self.z = self.spec.get("_z", -1.0)
        self.half = max(45.0, 5.0 * self.scene_extent) * lerp(0.8, 1.6, self.size)
        self.step = self.spec.get("_step", 2.0)

    def update(self, f):
        o = self.origin_at(f)
        self.r.floor = {"center": (float(o[0]), float(o[1]) + self.half * 0.4), "z": self.z, "half": self.half,
                        "step": self.step, "drive": 0.08 + 0.3 * self.c("glow", f)}


COMPONENTS = {
    "particle_field": ParticleField, "pillar_grid": PillarGrid, "ring_tunnel": RingTunnel,
    "hero_sphere": HeroSphere, "wire_cage": WireCage, "shard_swarm": ShardSwarm, "tower_rows": TowerRows,
    "double_helix": DoubleHelix, "halo_ring": HaloRing, "sun_disc": SunDisc, "dust": Dust, "grid_floor": GridFloor,
}
