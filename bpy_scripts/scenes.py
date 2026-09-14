"""Scenes: geometry, motion, staging, camera and audio response.

Each scene has a build(scn, mat, sig, intensity) and update(f), is staged in
depth (hero / field / backdrop), and gives different parts of the music to
different layers: bass moves mass, mids drive rotation and flow, highs and
onsets sparkle. Scenes write the drive contract only -- never colours.
"""
import math

from mathutils import Vector

from kit import (CameraRig, Dust, FloorGrid, Halo, Shot, box_mesh, disc_mesh, fixed, ico_mesh, new_empty,
                 new_obj, orbit, ring_mesh, set_drive, tilt_matrix, wireframe)
from signals import clamp01, hash01


class RippleField:
    """Sphere field carrying bass-driven ripples; each beat launches a ring
    pulse that flashes in the accent colour. Ringed by a halo, over a grid floor."""

    N, SPACING, WAVELENGTH = 18, 0.55, 6.0

    def build(self, scn, mat, sig, k):
        self.sig, self.k = sig, k
        mesh = ico_mesh("sphere", 0.16, 1)
        half = (self.N - 1) / 2
        self.objs = []
        for ix in range(self.N):
            for iy in range(self.N):
                x, y = (ix - half) * self.SPACING, (iy - half) * self.SPACING
                self.objs.append((new_obj(scn, f"sph_{ix}_{iy}", mesh, mat, (x, y, 0)), math.hypot(x, y), ix * self.N + iy))
        self.phase = sig.cumulative(lambda f: 0.3 + 0.9 * sig.mid[f] * k)
        self.floor = FloorGrid(scn, mat, sig, size=90, cells=45, z=-2.2)
        self.halo = Halo(scn, mat, sig, radius=7.2, tube=0.05, loc=(0, 0, -1.2))
        self.dust = Dust(scn, mat, sig, count=140)
        self.rig = CameraRig(scn, sig, [
            Shot(24, orbit(10.0, 1.6, math.radians(-50), 0.04), fixed(0, 0, -0.4), shift=(0.14, 0.0)),
            Shot(85, orbit(38.0, 22.0, math.radians(20), 0.02), fixed(0, 0, -0.5)),
            Shot(18, fixed(0.3, -4.2, 0.9, drift=(0, 0.35, 0)), fixed(0, 8, 0.3, drift=(0, 0.35, 0)), roll=8),
            Shot(50, fixed(2.0, -4.0, 16.0), fixed(0, 0, 0), roll=28, shift=(0.0, -0.1)),
        ])

    def update(self, f):
        sig, k = self.sig, self.k
        amp = 0.1 + 1.0 * sig.low[f] * k
        pulse_r = sig.since_beat[f] * 7.0
        pulse_h = 0.9 * k * sig.beat[f]
        span = amp + pulse_h + 1e-6
        b = sig.beat_index[f]
        for o, dist, i in self.objs:
            ring = math.exp(-((dist - pulse_r) / 0.7) ** 2)
            z = amp * math.sin(2 * math.pi * (self.phase[f] - dist / self.WAVELENGTH)) + pulse_h * ring
            o.location.z = z
            drive = 0.5 + 0.5 * z / span
            if hash01(i, b, 11) > 0.9:
                drive += 0.5 * sig.high[f]
            s = 0.6 + 0.6 * clamp01(drive)
            o.scale = (s, s, s)
            set_drive(o, drive, ring * sig.beat[f] * 1.5)
        self.floor.update(f)
        self.halo.update(f, signal=sig.low)
        self.dust.update(f)
        self.rig.update(f)


class MonolithGrid:
    """Pillar grid on a grid floor: bass sets the shared height, each beat fires
    a random subset (accent on downbeats). Giant slabs stand on the horizon."""

    N, SPACING, FIRE_P = 12, 1.3, 0.2

    def build(self, scn, mat, sig, k):
        self.sig, self.k = sig, k
        mesh = box_mesh("pillar", 0.45 * 2, 0.45 * 2, 1.0)
        half = (self.N - 1) / 2
        self.objs = []
        for ix in range(self.N):
            for iy in range(self.N):
                i = ix * self.N + iy
                o = new_obj(scn, f"pil_{ix}_{iy}", mesh, mat, ((ix - half) * self.SPACING, (iy - half) * self.SPACING, 0))
                self.objs.append((o, i, 0.6 + 1.2 * hash01(i, 1), 6.28 * hash01(i, 2)))
        slab = box_mesh("slab", 5.0, 3.0, 1.0)
        self.slabs = []
        for j in range(7):
            x = -42 + j * 14 + 4 * (hash01(j, 5) - 0.5)
            o = new_obj(scn, f"slab_{j}", slab, mat, (x, 55 + 10 * hash01(j, 6), 0))
            o.scale.z = 22 + 20 * hash01(j, 7)
            self.slabs.append(o)
        self.floor = FloorGrid(scn, mat, sig, size=140, cells=70, z=0.0, center=(0, 20))
        self.dust = Dust(scn, mat, sig, count=120, r_min=25, r_max=60, z_min=3, z_max=40)
        self.rig = CameraRig(scn, sig, [
            Shot(20, fixed(0.0, -10.0, 0.6, drift=(0, 0.35, 0)), fixed(0.0, 12.0, 2.8, drift=(0, 0.35, 0)), roll=-6),
            Shot(85, orbit(46.0, 34.0, math.radians(-60), 0.02), fixed(0, 0, 1.0)),
            Shot(35, fixed(13.0, -11.0, 4.0), fixed(0, 0, 1.5), shift=(-0.16, 0.0)),
            Shot(24, fixed(-9.0, -15.0, 1.0, drift=(0.2, 0.1, 0)), fixed(0, 25, 7), shift=(0.1, 0.12)),
        ])

    def update(self, f):
        sig, k = self.sig, self.k
        t, b, bi = sig.t[f], sig.beat[f], sig.beat_index[f]
        max_h = 0.15 + 1.2 * k + 3.5 * k
        downbeat = bi >= 0 and bi % 4 == 0
        for o, i, w, ph in self.objs:
            h = 0.15 + 1.2 * k * sig.low[f] * (0.25 + 0.75 * (0.5 + 0.5 * math.sin(t * w + ph)))
            fired = bi >= 0 and hash01(bi, i, 3) < self.FIRE_P
            if fired:
                h += 3.5 * k * b
            o.scale.z = max(h, 0.02)
            set_drive(o, h / max_h + 0.3 * sig.high[f] * hash01(i, bi, 9), 1.0 if fired and downbeat else 0.0)
        for j, o in enumerate(self.slabs):
            set_drive(o, 0.04 + 0.35 * sig.low[f] * (0.6 + 0.4 * hash01(j, 8)))
        self.floor.update(f)
        self.dust.update(f)
        self.rig.update(f)


class Tunnel:
    """Flight through hexagonal rings: speed follows energy, mids twist the
    rings, beats send a pulse down the tunnel, highs flicker the wall streaks."""

    RINGS, SPACING, RADIUS = 40, 3.0, 4.5

    def build(self, scn, mat, sig, k):
        self.sig, self.k = sig, k
        length = self.RINGS * self.SPACING
        self.length = length
        self.travel = sig.cumulative(lambda f: (3.0 + 10.0 * sig.energy[f]) * (0.5 + 0.5 * k))
        self.twist = sig.cumulative(lambda f: 0.15 + 1.3 * sig.mid[f] * k)
        ring = ring_mesh("tunnel_ring", self.RADIUS, 0.12, 6, 4)
        self.rings = [new_obj(scn, f"ring_{i}", ring, mat, accent=1.0 if i % 4 == 0 else 0.0) for i in range(self.RINGS)]
        streak = box_mesh("streak", 0.05, 1.8, 0.05, base=False)
        self.streaks = []
        for i in range(64):
            a = 2 * math.pi * hash01(i, 21)
            r = self.RADIUS * (0.75 + 0.2 * hash01(i, 22))
            self.streaks.append((new_obj(scn, f"streak_{i}", streak, mat), a, r, length * hash01(i, 23), i))
        self.exit = new_obj(scn, "exit", disc_mesh("exit", 2.5), mat)
        self.rig = CameraRig(scn, sig, [
            Shot(18, lambda f, u: (0.0, self.travel[f], 0.0), lambda f, u: (0.0, self.travel[f] + 15, 0.0)),
            Shot(28, lambda f, u: (2.2, self.travel[f], -1.8), lambda f, u: (0.4, self.travel[f] + 20, 0.4), roll=14),
            Shot(12, lambda f, u: (0.0, self.travel[f], 1.6), lambda f, u: (0.0, self.travel[f] + 10, 1.6), roll=-20),
        ], clip_end=length + 20)

    def update(self, f):
        sig, k = self.sig, self.k
        cam_y = self.travel[f]
        front = sig.since_beat[f] * 28.0
        for i, o in enumerate(self.rings):
            rel = (i * self.SPACING - cam_y) % self.length - 2.0
            bump = math.exp(-((rel - front) / 3.0) ** 2) * sig.beat[f]
            s = 1.0 + 0.2 * sig.low[f] * k + 0.3 * bump * k
            o.location = (0.0, cam_y + rel, 0.0)
            o.rotation_euler = (math.pi / 2, self.twist[f] + i * 0.09, 0.0)
            o.scale = (s, s, s)
            set_drive(o, 0.1 + 0.9 * bump + 0.25 * sig.high[f] * hash01(i, sig.beat_index[f], 4),
                      1.0 if i % 4 == 0 else 0.0)
        for o, a, r, offset, i in self.streaks:
            rel = (offset - cam_y * 1.6) % self.length
            o.location = (r * math.cos(a), cam_y + rel, r * math.sin(a))
            set_drive(o, sig.high[f] * (1.0 if hash01(i, sig.beat_index[f], 5) > 0.5 else 0.2))
        self.exit.location = (0.0, cam_y + self.length - 4.0, 0.0)
        set_drive(self.exit, 0.5 + 0.5 * sig.energy[f])
        self.rig.update(f)


class OrbitalCore:
    """Hero core breathing with the bass inside a counter-rotating wire cage,
    three tilted shard orbits whose speed follows the mids, and a halo that
    flashes on beats."""

    ORBITS = [(3.8, 30, (20, 0, 0)), (5.5, 40, (-35, 0, 40)), (7.5, 50, (60, 0, -30))]

    def build(self, scn, mat, sig, k):
        self.sig, self.k = sig, k
        self.core = new_obj(scn, "core", ico_mesh("core", 1.4, 3), mat)
        self.cage = wireframe(new_obj(scn, "cage", ico_mesh("cage", 2.1, 1), mat, accent=1.0), 0.03)
        self.spin = sig.cumulative(lambda f: 0.1 + 0.6 * sig.mid[f] * k)
        self.orbit_phase = sig.cumulative(lambda f: 0.12 + 0.8 * sig.mid[f] * k)
        shard = ico_mesh("shard", 0.14, 0)
        self.shards = []
        for oi, (radius, count, tilt) in enumerate(self.ORBITS):
            m = tilt_matrix(*tilt)
            for j in range(count):
                o = new_obj(scn, f"shard_{oi}_{j}", shard, mat, accent=1.0 if oi == 2 else 0.0)
                o.scale = (0.6 + 1.4 * hash01(oi, j, 1), 0.6 + 1.4 * hash01(oi, j, 2), 0.6 + 1.4 * hash01(oi, j, 3))
                o.rotation_euler = (6.28 * hash01(oi, j, 4), 6.28 * hash01(oi, j, 5), 0)
                r = radius * (0.92 + 0.16 * hash01(oi, j, 6))
                self.shards.append((o, m, r, 2 * math.pi * j / count, 3.8 / radius, oi * 100 + j, oi))
        self.halo = Halo(scn, mat, sig, radius=11.0, tube=0.05, rot_deg=(75, 0, 20))
        self.dust = Dust(scn, mat, sig, count=160, r_min=18, r_max=50, z_min=-20, z_max=20)
        self.rig = CameraRig(scn, sig, [
            Shot(35, orbit(7.8, 1.0, math.radians(-70), 0.08), fixed(0, 0, 0), shift=(-0.2, 0.0)),
            Shot(16, fixed(0.0, -15.0, 5.0), fixed(0, 0, 0.5), roll=10),
            Shot(135, orbit(44.0, 4.0, math.radians(30), 0.01), fixed(0, 0, 0)),
            Shot(24, orbit(5.0, -1.5, math.radians(200), 0.12), fixed(0, 0, 1.0), roll=-12),
        ])

    def update(self, f):
        sig, k = self.sig, self.k
        s = 1.0 + 0.35 * k * sig.low[f] + 0.2 * k * sig.beat[f]
        self.core.scale = (s, s, s)
        self.core.rotation_euler = (0.3 * self.spin[f], 0.0, self.spin[f])
        set_drive(self.core, 0.12 + 0.55 * sig.low[f])
        c = 1.0 + 0.15 * k * sig.mid[f]
        self.cage.scale = (c, c, c)
        self.cage.rotation_euler = (0.0, -0.4 * self.spin[f], -self.spin[f])
        set_drive(self.cage, 0.15 + 0.7 * sig.mid[f], 1.0)
        b = sig.beat_index[f]
        phase = self.orbit_phase[f]
        for o, m, r, a0, speed, i, oi in self.shards:
            a = a0 + phase * speed
            o.location = m @ Vector((r * math.cos(a), r * math.sin(a), 0.0))
            flash = sig.onset[f] if hash01(i, b, 2) > 0.7 else 0.0
            set_drive(o, 0.08 + 0.6 * sig.high[f] * hash01(i, b, 3) + 0.8 * flash, 1.0 if oi == 2 else 0.0)
        self.halo.update(f)
        self.dust.update(f)
        self.rig.update(f)


class SpectrumStreet:
    """Two rows of towers receding to a horizon sun. Each tower replays the
    audio from a moment ago, so the music's recent history rolls away into the
    distance. Left row bass, right row mids/highs (accent)."""

    TOWERS, SPACING, DELAY_S = 44, 2.2, 0.05

    def build(self, scn, mat, sig, k):
        self.sig, self.k = sig, k
        tower = box_mesh("tower", 1.0, 1.0, 1.0)
        self.rows = []
        for side, x in ((0, -3.2), (1, 3.2)):
            for j in range(self.TOWERS):
                self.rows.append((new_obj(scn, f"tower_{side}_{j}", tower, mat, (x, 3.0 + j * self.SPACING, 0),
                                         accent=float(side)), side, j))
        dash = box_mesh("dash", 0.12, 1.0, 0.02)
        self.dashes = [(new_obj(scn, f"dash_{j}", dash, mat, (0, 3.0 + j * self.SPACING, 0)), j)
                       for j in range(self.TOWERS)]
        self.sun = new_obj(scn, "sun", disc_mesh("sun", 26.0), mat, (0, 175, 20))
        self.floor = FloorGrid(scn, mat, sig, size=260, cells=65, z=-0.01, center=(0, 100))
        self.dust = Dust(scn, mat, sig, count=80, r_min=40, r_max=90, z_min=15, z_max=60, center=(0, 60, 0))
        self.rig = CameraRig(scn, sig, [
            Shot(24, fixed(0.0, -8.0, 1.1, drift=(0, 0.5, 0)), fixed(0, 40, 3.5, drift=(0, 0.5, 0)), shift=(0.0, 0.08)),
            Shot(45, fixed(10.0, 6.0, 2.5, drift=(0, 0.3, 0)), fixed(-1, 30, 3), shift=(0.12, 0.0)),
            Shot(35, fixed(0.0, -12.0, 14.0), fixed(0, 35, 0)),
            Shot(85, fixed(-2.0, -40.0, 3.0), fixed(0, 60, 6)),
        ], clip_end=500)

    def update(self, f):
        sig, k = self.sig, self.k
        delay = self.DELAY_S * sig.fps
        top = 0.2 + 7.0 * k
        for o, side, j in self.rows:
            h_f = f - int(j * delay)
            level = sig.at(sig.low, h_f) if side == 0 else 0.5 * (sig.at(sig.mid, h_f) + sig.at(sig.high, h_f))
            h = 0.2 + 7.0 * k * level
            o.scale.z = h
            set_drive(o, h / top, float(side))
        for o, j in self.dashes:
            set_drive(o, 0.05 + 0.9 * sig.at(sig.beat, f - int(j * delay)))
        set_drive(self.sun, 0.55 + 0.45 * sig.energy[f])
        self.floor.update(f)
        self.dust.update(f)
        self.rig.update(f)


class Helix:
    """Rotating double helix with rungs: energy spins it, beats send pulses
    climbing the strands, bass swells the stacked halos around it."""

    PER_STRAND, RADIUS, RISE = 64, 2.4, 0.38

    def build(self, scn, mat, sig, k):
        self.sig, self.k = sig, k
        self.pivot = new_empty(scn, "helix")
        self.height = self.PER_STRAND * self.RISE
        self.spin = sig.cumulative(lambda f: 0.25 + 1.2 * sig.energy[f] * k)
        sph = ico_mesh("helix_sphere", 0.22, 2)
        self.spheres = []
        for strand in range(2):
            for j in range(self.PER_STRAND):
                a = j * 0.35 + strand * math.pi
                loc = (self.RADIUS * math.cos(a), self.RADIUS * math.sin(a), j * self.RISE)
                self.spheres.append((new_obj(scn, f"hx_{strand}_{j}", sph, mat, loc, accent=float(strand),
                                             parent=self.pivot), strand, j))
        rung = box_mesh("rung", 2 * self.RADIUS, 0.06, 0.06, base=False)
        self.rungs = []
        for j in range(0, self.PER_STRAND, 4):
            o = new_obj(scn, f"rung_{j}", rung, mat, (0, 0, j * self.RISE), parent=self.pivot)
            o.rotation_euler.z = j * 0.35
            self.rungs.append(o)
        halo = ring_mesh("helix_halo", 5.0, 0.05, 96, 6)
        self.halos = [new_obj(scn, f"hhalo_{i}", halo, mat, (0, 0, 3 + i * 6.0)) for i in range(4)]
        self.floor = FloorGrid(scn, mat, sig, size=90, cells=45, z=-0.6)
        self.dust = Dust(scn, mat, sig, count=150, r_min=15, r_max=40, z_min=-5, z_max=40)
        mid = self.height / 2
        self.rig = CameraRig(scn, sig, [
            Shot(28, lambda f, u: (9 * math.cos(-1.2 + 0.05 * u), 9 * math.sin(-1.2 + 0.05 * u), 3 + 1.5 * u),
                 lambda f, u: (0, 0, 6 + 1.5 * u), shift=(0.15, 0.0)),
            Shot(18, fixed(1.5, -2.8, -0.2), fixed(0, 0, 14), roll=15),
            Shot(100, fixed(0.0, -48.0, mid), fixed(0, 0, mid), shift=(0.18, 0.0)),
            Shot(35, fixed(0.0, -3.0, self.height + 10), fixed(0, 0, mid), roll=30),
        ])

    def update(self, f):
        sig, k = self.sig, self.k
        self.pivot.rotation_euler.z = self.spin[f]
        front = (sig.since_beat[f] * 20.0) % (self.height + 6)
        b = sig.beat_index[f]
        for o, strand, j in self.spheres:
            z = j * self.RISE
            bump = math.exp(-((z - front) / 1.2) ** 2) * sig.beat[f]
            s = 1.0 + 1.0 * bump * k + 0.3 * sig.low[f] * k
            o.scale = (s, s, s)
            set_drive(o, 0.1 + 0.8 * bump + 0.3 * sig.high[f] * hash01(strand, j, b), float(strand))
        for o in self.rungs:
            set_drive(o, 0.05 + 0.6 * sig.mid[f], 0.5)
        for i, o in enumerate(self.halos):
            s = 1.0 + 0.25 * sig.low[f] * k * (1.0 - 0.15 * i)
            o.scale = (s, s, 1.0)
            set_drive(o, 0.08 + 0.6 * sig.low[f], 1.0)
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
