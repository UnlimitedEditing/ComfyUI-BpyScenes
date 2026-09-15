"""Music visualizer scenes for the GL renderer -- vectorised ports of
bpy_scripts/scenes.py (same names, motion, staging and shot lists).

build(r, sig, k) creates batches on the Renderer; update(f) sets numpy arrays
on them plus r.camera / r.floor / r.hero. Scenes never set colours: they write
drive (0-1 excitement) and accent (0 palette, 1 accent) per instance.
"""
import math

import numpy as np

from gpu import box_mesh, disc_mesh, hash01, ico_mesh, model_matrices, ring_mesh, wire_mesh


def _rot_z(points, angle):
    c, s = math.cos(angle), math.sin(angle)
    out = points.copy()
    out[:, 0], out[:, 1] = c * points[:, 0] - s * points[:, 1], s * points[:, 0] + c * points[:, 1]
    return out


def _tilt(rx, ry, rz):
    return model_matrices([[0, 0, 0]], rot=[[math.radians(rx), math.radians(ry), math.radians(rz)]])[0] \
        .reshape(4, 4).T[:3, :3]


# ── shared staging ───────────────────────────────────────────────────────────

class Shot:
    def __init__(self, lens, pos, target, shift=(0.0, 0.0), roll=0.0, shake=1.0):
        self.lens, self.pos, self.target, self.shift, self.roll, self.shake = lens, pos, target, shift, roll, shake


def orbit(radius, z, angle0=0.0, speed=0.05, center=(0.0, 0.0)):
    return lambda f, u: (center[0] + radius * math.cos(angle0 + speed * u),
                         center[1] + radius * math.sin(angle0 + speed * u), z)


def fixed(x, y, z, drift=(0.0, 0.0, 0.0)):
    return lambda f, u: (x + drift[0] * u, y + drift[1] * u, z + drift[2] * u)


class CameraRig:
    """Hard cut to the next shot at every sig.cut change; small shake on bass hits."""

    def __init__(self, r, sig, shots, clip_far=500.0):
        self.r, self.sig, self.shots = r, sig, shots
        r.clip_far = clip_far

    def update(self, f):
        sig = self.sig
        shot = self.shots[sig.cut[f] % len(self.shots)]
        u = sig.since_cut[f]
        eye = np.array(shot.pos(f, u), dtype=np.float64)
        target = np.array(shot.target(f, u), dtype=np.float64)
        amp = 0.06 * shot.shake * (0.5 * sig.low[f] * sig.beat[f] + 0.5 * sig.onset[f])
        t = sig.t[f]
        eye += np.array([math.sin(t * 37.0), math.sin(t * 29.0 + 1.3), math.sin(t * 41.0 + 2.1)]) * amp
        self.r.camera = {"eye": eye, "target": target, "lens": shot.lens, "shift": shot.shift, "roll": shot.roll}


class Floor:
    def __init__(self, r, sig, half=45.0, step=2.0, z=0.0, center=(0.0, 0.0)):
        self.r, self.sig = r, sig
        self.spec = {"center": center, "z": z, "half": half, "step": step, "drive": 0.0}

    def update(self, f, base=0.08, gain=0.25):
        self.spec["drive"] = base + gain * self.sig.onset[f]
        self.r.floor = self.spec


class Halo:
    def __init__(self, r, sig, radius, tube=0.06, loc=(0, 0, 0), rot_deg=(0, 0, 0), accent=1.0):
        self.sig = sig
        self.b = r.batch(ring_mesh(radius, tube, 96, 6), 1)
        self.b.pos[:] = loc
        self.b.rot = np.radians([rot_deg])
        self.b.accent[:] = accent

    def update(self, f, base=0.12, gain=0.7, signal=None):
        s = (signal if signal is not None else self.sig.beat)[f]
        self.b.drive[:] = base + gain * s


class Dust:
    """Sparse particles on a distant slowly spinning shell; hi-hat sparkle."""

    def __init__(self, r, sig, count=140, r_min=20.0, r_max=45.0, z_min=-5.0, z_max=25.0, center=(0, 0, 0), seed=3):
        self.sig = sig
        i = np.arange(count)
        a = 2 * np.pi * hash01(seed, i, 1)
        rad = r_min + (r_max - r_min) * hash01(seed, i, 2)
        self.base = np.stack([rad * np.cos(a), rad * np.sin(a), z_min + (z_max - z_min) * hash01(seed, i, 3)], 1)
        self.center = np.asarray(center, dtype=np.float64)
        self.b = r.batch(ico_mesh(0.07, 0)[0], count)
        s = 0.6 + 1.8 * hash01(seed, i, 4)
        self.b.scale[:] = np.stack([s, s, s], 1)
        self.idx = i
        self.spin = sig.cumulative(lambda f: 0.01 + 0.03 * sig.energy[f])

    def update(self, f):
        sig = self.sig
        self.b.pos[:] = _rot_z(self.base, self.spin[f]) + self.center
        flicker = sig.high[f] * np.where(hash01(self.idx, sig.beat_index[f], 7) > 0.6, 1.0, 0.25)
        self.b.drive[:] = 0.05 + 0.8 * flicker


# ── scenes ───────────────────────────────────────────────────────────────────

class RippleField:
    N, SPACING, WAVELENGTH = 18, 0.55, 6.0

    def build(self, r, sig, k):
        self.r, self.sig, self.k = r, sig, k
        half = (self.N - 1) / 2
        gx, gy = np.meshgrid((np.arange(self.N) - half) * self.SPACING, (np.arange(self.N) - half) * self.SPACING,
                             indexing="ij")
        self.xy = np.stack([gx.ravel(), gy.ravel()], 1)
        self.dist = np.hypot(self.xy[:, 0], self.xy[:, 1])
        self.idx = np.arange(self.N * self.N)
        self.b = r.batch(ico_mesh(0.16, 1)[0], self.N * self.N)
        self.phase = sig.cumulative(lambda f: 0.3 + 0.9 * sig.mid[f] * k)
        self.floor = Floor(r, sig, half=45, step=2.0, z=-2.2)
        self.halo = Halo(r, sig, radius=7.2, tube=0.05, loc=(0, 0, -1.2))
        self.dust = Dust(r, sig, count=140)
        self.rig = CameraRig(r, sig, [
            Shot(24, orbit(10.0, 1.6, math.radians(-50), 0.04), fixed(0, 0, -0.4), shift=(0.14, 0.0)),
            Shot(85, orbit(38.0, 22.0, math.radians(20), 0.02), fixed(0, 0, -0.5)),
            Shot(18, fixed(0.3, -4.2, 0.9, drift=(0, 0.35, 0)), fixed(0, 8, 0.3, drift=(0, 0.35, 0)), roll=8),
            Shot(50, fixed(2.0, -4.0, 16.0), fixed(0, 0, 0), roll=28, shift=(0.0, -0.1)),
        ])

    def update(self, f):
        sig, k = self.sig, self.k
        amp = 0.1 + 1.0 * sig.low[f] * k
        pulse_h = 0.9 * k * sig.beat[f]
        ring = np.exp(-((self.dist - sig.since_beat[f] * 7.0) / 0.7) ** 2)
        z = amp * np.sin(2 * np.pi * (self.phase[f] - self.dist / self.WAVELENGTH)) + pulse_h * ring
        drive = 0.5 + 0.5 * z / (amp + pulse_h + 1e-6)
        drive = drive + np.where(hash01(self.idx, sig.beat_index[f], 11) > 0.9, 0.5 * sig.high[f], 0.0)
        s = 0.6 + 0.6 * np.clip(drive, 0, 1)
        self.b.pos[:, :2], self.b.pos[:, 2] = self.xy, z
        self.b.scale[:] = s[:, None]
        self.b.drive, self.b.accent = drive, ring * sig.beat[f] * 1.5
        self.r.hero = (0.0, 0.0, 0.0)
        self.floor.update(f)
        self.halo.update(f, signal=sig.low)
        self.dust.update(f)
        self.rig.update(f)


class MonolithGrid:
    N, SPACING, FIRE_P = 12, 1.3, 0.2

    def build(self, r, sig, k):
        self.r, self.sig, self.k = r, sig, k
        half = (self.N - 1) / 2
        gx, gy = np.meshgrid((np.arange(self.N) - half) * self.SPACING, (np.arange(self.N) - half) * self.SPACING,
                             indexing="ij")
        self.idx = np.arange(self.N * self.N)
        self.b = r.batch(box_mesh(0.9, 0.9, 1.0), self.N * self.N)
        self.b.pos[:, 0], self.b.pos[:, 1] = gx.ravel(), gy.ravel()
        self.w = 0.6 + 1.2 * hash01(self.idx, 1)
        self.ph = 6.28 * hash01(self.idx, 2)
        j = np.arange(7)
        self.slabs = r.batch(box_mesh(5.0, 3.0, 1.0), 7)
        self.slabs.pos[:, 0] = -42 + j * 14 + 4 * (hash01(j, 5) - 0.5)
        self.slabs.pos[:, 1] = 55 + 10 * hash01(j, 6)
        self.slabs.scale[:, 2] = 22 + 20 * hash01(j, 7)
        self.slab_var = 0.6 + 0.4 * hash01(j, 8)
        self.floor = Floor(r, sig, half=70, step=2.0, z=0.0, center=(0, 20))
        self.dust = Dust(r, sig, count=120, r_min=25, r_max=60, z_min=3, z_max=40)
        self.rig = CameraRig(r, sig, [
            Shot(20, fixed(0.0, -10.0, 0.6, drift=(0, 0.35, 0)), fixed(0.0, 12.0, 2.8, drift=(0, 0.35, 0)), roll=-6),
            Shot(85, orbit(46.0, 34.0, math.radians(-60), 0.02), fixed(0, 0, 1.0)),
            Shot(35, fixed(13.0, -11.0, 4.0), fixed(0, 0, 1.5), shift=(-0.16, 0.0)),
            Shot(24, fixed(-9.0, -15.0, 1.0, drift=(0.2, 0.1, 0)), fixed(0, 25, 7), shift=(0.1, 0.12)),
        ])

    def update(self, f):
        sig, k = self.sig, self.k
        t, bi = sig.t[f], sig.beat_index[f]
        max_h = 0.15 + 1.2 * k + 3.5 * k
        h = 0.15 + 1.2 * k * sig.low[f] * (0.25 + 0.75 * (0.5 + 0.5 * np.sin(t * self.w + self.ph)))
        fired = (hash01(bi, self.idx, 3) < self.FIRE_P) if bi >= 0 else np.zeros(len(self.idx), bool)
        h = h + np.where(fired, 3.5 * k * sig.beat[f], 0.0)
        self.b.scale[:, 2] = np.maximum(h, 0.02)
        self.b.drive = h / max_h + 0.3 * sig.high[f] * hash01(self.idx, bi, 9)
        self.b.accent = np.where(fired & (bi >= 0 and bi % 4 == 0), 1.0, 0.0)
        self.slabs.drive = 0.04 + 0.35 * sig.low[f] * self.slab_var
        self.r.hero = (0.0, 55.0, 25.0)
        self.floor.update(f)
        self.dust.update(f)
        self.rig.update(f)


class Tunnel:
    RINGS, SPACING, RADIUS = 40, 3.0, 4.5

    def build(self, r, sig, k):
        self.r, self.sig, self.k = r, sig, k
        self.length = self.RINGS * self.SPACING
        self.travel = sig.cumulative(lambda f: (3.0 + 10.0 * sig.energy[f]) * (0.5 + 0.5 * k))
        self.twist = sig.cumulative(lambda f: 0.15 + 1.3 * sig.mid[f] * k)
        self.i = np.arange(self.RINGS)
        self.rings = r.batch(ring_mesh(self.RADIUS, 0.12, 6, 4), self.RINGS)
        self.rings.accent = np.where(self.i % 4 == 0, 1.0, 0.0)
        s = np.arange(64)
        self.s_angle = 2 * np.pi * hash01(s, 21)
        self.s_radius = self.RADIUS * (0.75 + 0.2 * hash01(s, 22))
        self.s_offset = self.length * hash01(s, 23)
        self.s_idx = s
        self.streaks = r.batch(box_mesh(0.05, 1.8, 0.05, base=False), 64)
        self.exit = r.batch(disc_mesh(2.5), 1)
        self.rig = CameraRig(r, sig, [
            Shot(18, lambda f, u: (0.0, self.travel[f], 0.0), lambda f, u: (0.0, self.travel[f] + 15, 0.0)),
            Shot(28, lambda f, u: (2.2, self.travel[f], -1.8), lambda f, u: (0.4, self.travel[f] + 20, 0.4), roll=14),
            Shot(12, lambda f, u: (0.0, self.travel[f], 1.6), lambda f, u: (0.0, self.travel[f] + 10, 1.6), roll=-20),
        ], clip_far=self.length + 20)

    def update(self, f):
        sig, k = self.sig, self.k
        cam_y = self.travel[f]
        rel = (self.i * self.SPACING - cam_y) % self.length - 2.0
        bump = np.exp(-((rel - sig.since_beat[f] * 28.0) / 3.0) ** 2) * sig.beat[f]
        s = 1.0 + 0.2 * sig.low[f] * k + 0.3 * bump * k
        self.rings.pos[:, 0], self.rings.pos[:, 1], self.rings.pos[:, 2] = 0.0, cam_y + rel, 0.0
        self.rings.rot = np.stack([np.full(self.RINGS, math.pi / 2), self.twist[f] + self.i * 0.09,
                                   np.zeros(self.RINGS)], 1)
        self.rings.scale[:] = s[:, None]
        self.rings.drive = 0.1 + 0.9 * bump + 0.25 * sig.high[f] * hash01(self.i, sig.beat_index[f], 4)
        srel = (self.s_offset - cam_y * 1.6) % self.length
        self.streaks.pos[:, 0] = self.s_radius * np.cos(self.s_angle)
        self.streaks.pos[:, 1] = cam_y + srel
        self.streaks.pos[:, 2] = self.s_radius * np.sin(self.s_angle)
        self.streaks.drive = sig.high[f] * np.where(hash01(self.s_idx, sig.beat_index[f], 5) > 0.5, 1.0, 0.2)
        self.exit.pos[:] = (0.0, cam_y + self.length - 4.0, 0.0)
        self.exit.drive[:] = 0.5 + 0.5 * sig.energy[f]
        self.r.hero = (0.0, cam_y + self.length - 4.0, 0.0)
        self.r.floor = None
        self.rig.update(f)


class OrbitalCore:
    ORBITS = [(3.8, 30, (20, 0, 0)), (5.5, 40, (-35, 0, 40)), (7.5, 50, (60, 0, -30))]

    def build(self, r, sig, k):
        self.r, self.sig, self.k = r, sig, k
        self.core = r.batch(ico_mesh(1.4, 3)[0], 1)
        _, cv, cf = ico_mesh(2.1, 1)
        self.cage = r.batch(wire_mesh(cv, cf, 0.03), 1)
        self.cage.accent[:] = 1.0
        self.spin = sig.cumulative(lambda f: 0.1 + 0.6 * sig.mid[f] * k)
        self.orbit_phase = sig.cumulative(lambda f: 0.12 + 0.8 * sig.mid[f] * k)
        total = sum(c for _, c, _ in self.ORBITS)
        self.shards = r.batch(ico_mesh(0.14, 0)[0], total)
        radii, a0, speed, tilt_idx, ids, orbit_id = [], [], [], [], [], []
        for oi, (radius, count, tilt) in enumerate(self.ORBITS):
            j = np.arange(count)
            radii.append(radius * (0.92 + 0.16 * hash01(oi, j, 6)))
            a0.append(2 * np.pi * j / count)
            speed.append(np.full(count, 3.8 / radius))
            ids.append(oi * 100 + j)
            orbit_id.append(np.full(count, oi))
        self.radii, self.a0, self.speed = np.concatenate(radii), np.concatenate(a0), np.concatenate(speed)
        self.ids, self.orbit_id = np.concatenate(ids), np.concatenate(orbit_id)
        self.tilts = np.stack([_tilt(*t) for _, _, t in self.ORBITS])[self.orbit_id]
        oid, jid = self.orbit_id, self.ids % 100
        self.shards.scale[:] = np.stack([0.6 + 1.4 * hash01(oid, jid, n) for n in (1, 2, 3)], 1)
        self.shards.rot = np.stack([6.28 * hash01(oid, jid, 4), 6.28 * hash01(oid, jid, 5), np.zeros(total)], 1)
        self.shards.accent = np.where(oid == 2, 1.0, 0.0)
        self.halo = Halo(r, sig, radius=11.0, tube=0.05, rot_deg=(75, 0, 20))
        self.dust = Dust(r, sig, count=160, r_min=18, r_max=50, z_min=-20, z_max=20)
        self.rig = CameraRig(r, sig, [
            Shot(35, orbit(7.8, 1.0, math.radians(-70), 0.08), fixed(0, 0, 0), shift=(-0.2, 0.0)),
            Shot(16, fixed(0.0, -15.0, 5.0), fixed(0, 0, 0.5), roll=10),
            Shot(135, orbit(44.0, 4.0, math.radians(30), 0.01), fixed(0, 0, 0)),
            Shot(24, orbit(5.0, -1.5, math.radians(200), 0.12), fixed(0, 0, 1.0), roll=-12),
        ])

    def update(self, f):
        sig, k = self.sig, self.k
        s = 1.0 + 0.35 * k * sig.low[f] + 0.2 * k * sig.beat[f]
        self.core.scale[:] = s
        self.core.rot = np.array([[0.3 * self.spin[f], 0.0, self.spin[f]]])
        self.core.drive[:] = 0.12 + 0.55 * sig.low[f]
        c = 1.0 + 0.15 * k * sig.mid[f]
        self.cage.scale[:] = c
        self.cage.rot = np.array([[0.0, -0.4 * self.spin[f], -self.spin[f]]])
        self.cage.drive[:] = 0.15 + 0.7 * sig.mid[f]
        a = self.a0 + self.orbit_phase[f] * self.speed
        local = np.stack([self.radii * np.cos(a), self.radii * np.sin(a), np.zeros_like(a)], 1)
        self.shards.pos[:] = np.einsum("nij,nj->ni", self.tilts, local)
        b = sig.beat_index[f]
        flash = np.where(hash01(self.ids, b, 2) > 0.7, sig.onset[f], 0.0)
        self.shards.drive = 0.08 + 0.6 * sig.high[f] * hash01(self.ids, b, 3) + 0.8 * flash
        self.r.hero = (0.0, 0.0, 0.0)
        self.r.floor = None
        self.halo.update(f)
        self.dust.update(f)
        self.rig.update(f)


class SpectrumStreet:
    TOWERS, SPACING, DELAY_S = 44, 2.2, 0.05

    def build(self, r, sig, k):
        self.r, self.sig, self.k = r, sig, k
        j = np.arange(self.TOWERS)
        self.towers = r.batch(box_mesh(1.0, 1.0, 1.0), self.TOWERS * 2)
        self.towers.pos[:, 0] = np.concatenate([np.full(self.TOWERS, -3.2), np.full(self.TOWERS, 3.2)])
        self.towers.pos[:, 1] = np.tile(3.0 + j * self.SPACING, 2)
        self.towers.accent = np.concatenate([np.zeros(self.TOWERS), np.ones(self.TOWERS)])
        self.delay = (j * self.DELAY_S * sig.fps).astype(int)
        self.dashes = r.batch(box_mesh(0.12, 1.0, 0.02), self.TOWERS)
        self.dashes.pos[:, 1] = 3.0 + j * self.SPACING
        self.sun = r.batch(disc_mesh(26.0), 1)
        self.sun.pos[:] = (0, 175, 20)
        self.floor = Floor(r, sig, half=130, step=4.0, z=-0.01, center=(0, 100))
        self.dust = Dust(r, sig, count=80, r_min=40, r_max=90, z_min=15, z_max=60, center=(0, 60, 0))
        self.low, self.midhigh = np.asarray(sig.low), 0.5 * (np.asarray(sig.mid) + np.asarray(sig.high))
        self.beat = np.asarray(sig.beat)
        self.rig = CameraRig(r, sig, [
            Shot(24, fixed(0.0, -8.0, 1.1, drift=(0, 0.5, 0)), fixed(0, 40, 3.5, drift=(0, 0.5, 0)), shift=(0.0, 0.08)),
            Shot(45, fixed(10.0, 6.0, 2.5, drift=(0, 0.3, 0)), fixed(-1, 30, 3), shift=(0.12, 0.0)),
            Shot(35, fixed(0.0, -12.0, 14.0), fixed(0, 35, 0)),
            Shot(85, fixed(-2.0, -40.0, 3.0), fixed(0, 60, 6)),
        ], clip_far=500)

    def update(self, f):
        sig, k = self.sig, self.k
        hist = np.clip(f - self.delay, 0, sig.n - 1)
        top = 0.2 + 7.0 * k
        h = 0.2 + 7.0 * k * np.concatenate([self.low[hist], self.midhigh[hist]])
        self.towers.scale[:, 2] = h
        self.towers.drive = h / top
        self.dashes.drive = 0.05 + 0.9 * self.beat[hist]
        self.sun.drive[:] = 0.55 + 0.45 * sig.energy[f]
        self.r.hero = (0.0, 175.0, 20.0)
        self.floor.update(f)
        self.dust.update(f)
        self.rig.update(f)


class Helix:
    PER_STRAND, RADIUS, RISE = 64, 2.4, 0.38

    def build(self, r, sig, k):
        self.r, self.sig, self.k = r, sig, k
        self.height = self.PER_STRAND * self.RISE
        self.spin = sig.cumulative(lambda f: 0.25 + 1.2 * sig.energy[f] * k)
        j = np.arange(self.PER_STRAND)
        strand = np.concatenate([np.zeros(self.PER_STRAND), np.ones(self.PER_STRAND)])
        jj = np.tile(j, 2)
        a = jj * 0.35 + strand * math.pi
        self.local = np.stack([self.RADIUS * np.cos(a), self.RADIUS * np.sin(a), jj * self.RISE], 1)
        self.strand, self.jj = strand, jj
        self.spheres = r.batch(ico_mesh(0.22, 2)[0], self.PER_STRAND * 2)
        self.spheres.accent = strand
        rj = np.arange(0, self.PER_STRAND, 4)
        self.rung_z, self.rung_a = rj * self.RISE, rj * 0.35
        self.rungs = r.batch(box_mesh(2 * self.RADIUS, 0.06, 0.06, base=False), len(rj))
        self.rungs.accent[:] = 0.5
        self.halos = r.batch(ring_mesh(5.0, 0.05, 96, 6), 4)
        self.halos.pos[:, 2] = 3 + np.arange(4) * 6.0
        self.halos.accent[:] = 1.0
        self.floor = Floor(r, sig, half=45, step=2.0, z=-0.6)
        self.dust = Dust(r, sig, count=150, r_min=15, r_max=40, z_min=-5, z_max=40)
        mid = self.height / 2
        self.rig = CameraRig(r, sig, [
            Shot(28, lambda f, u: (9 * math.cos(-1.2 + 0.05 * u), 9 * math.sin(-1.2 + 0.05 * u), 3 + 1.5 * u),
                 lambda f, u: (0, 0, 6 + 1.5 * u), shift=(0.15, 0.0)),
            Shot(18, fixed(1.5, -2.8, -0.2), fixed(0, 0, 14), roll=15),
            Shot(100, fixed(0.0, -48.0, mid), fixed(0, 0, mid), shift=(0.18, 0.0)),
            Shot(35, fixed(0.0, -3.0, self.height + 10), fixed(0, 0, mid), roll=30),
        ])

    def update(self, f):
        sig, k = self.sig, self.k
        spin = self.spin[f]
        self.spheres.pos[:] = _rot_z(self.local, spin)
        front = (sig.since_beat[f] * 20.0) % (self.height + 6)
        bump = np.exp(-((self.local[:, 2] - front) / 1.2) ** 2) * sig.beat[f]
        s = 1.0 + 1.0 * bump * k + 0.3 * sig.low[f] * k
        self.spheres.scale[:] = s[:, None]
        self.spheres.drive = 0.1 + 0.8 * bump + 0.3 * sig.high[f] * hash01(self.strand, self.jj, sig.beat_index[f])
        self.rungs.pos[:, 2] = self.rung_z
        self.rungs.rot = np.stack([np.zeros_like(self.rung_a), np.zeros_like(self.rung_a), self.rung_a + spin], 1)
        self.rungs.drive[:] = 0.05 + 0.6 * sig.mid[f]
        hs = 1.0 + 0.25 * sig.low[f] * k * (1.0 - 0.15 * np.arange(4))
        self.halos.scale[:, 0], self.halos.scale[:, 1] = hs, hs
        self.halos.drive[:] = 0.08 + 0.6 * sig.low[f]
        self.r.hero = (0.0, 0.0, self.height * 0.6)
        self.floor.update(f)
        self.dust.update(f)
        self.rig.update(f)


SCENES = {
    "ripple_field": RippleField,
    "monolith_grid": MonolithGrid,
    "tunnel": Tunnel,
    "orbital_core": OrbitalCore,
    "spectrum_street": SpectrumStreet,
    "helix": Helix,
}
