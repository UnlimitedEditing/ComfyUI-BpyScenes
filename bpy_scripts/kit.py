"""Composition kit shared by scenes: meshes, depth layers, and a camera rig.

Scenes stay flat without staging, not without post effects: a hero element,
a mid-ground field and a backdrop at real depth, shot through varied lenses
with off-centre framing and hard cuts. Everything here writes the drive
contract (obj.color[0] drive, obj.color[1] accent) and never sets colours.
"""
import math

import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector

from signals import clamp01, hash01


# ── elements & meshes ────────────────────────────────────────────────────────

def set_drive(obj, drive, accent=0.0):
    obj.color = (clamp01(drive), clamp01(accent), 0.0, 1.0)


def new_obj(scn, name, mesh, mat, loc=(0, 0, 0), drive=0.0, accent=0.0, parent=None):
    if mat.name not in [m.name for m in mesh.materials]:
        mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = loc
    if parent is not None:
        obj.parent = parent
    scn.collection.objects.link(obj)
    set_drive(obj, drive, accent)
    return obj


def new_empty(scn, name, loc=(0, 0, 0), parent=None):
    e = bpy.data.objects.new(name, None)
    e.location = loc
    if parent is not None:
        e.parent = parent
    scn.collection.objects.link(e)
    return e


def _mesh_from_bmesh(name, bm):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def ico_mesh(name, radius, subdiv=1):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=radius)
    return _mesh_from_bmesh(name, bm)


def box_mesh(name, w, d, h, base=True):
    """Box; with base=True the origin sits on the bottom face so scale.z grows it upward."""
    z0 = 0.0 if base else -h / 2
    verts = [(-w / 2, -d / 2, z0), (w / 2, -d / 2, z0), (w / 2, d / 2, z0), (-w / 2, d / 2, z0),
             (-w / 2, -d / 2, z0 + h), (w / 2, -d / 2, z0 + h), (w / 2, d / 2, z0 + h), (-w / 2, d / 2, z0 + h)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    return mesh


def ring_mesh(name, radius, tube, segments=48, sides=6):
    """Torus in the XY plane (axis = Z)."""
    verts, faces = [], []
    for i in range(segments):
        a = 2 * math.pi * i / segments
        for j in range(sides):
            b = 2 * math.pi * j / sides
            r = radius + tube * math.cos(b)
            verts.append((r * math.cos(a), r * math.sin(a), tube * math.sin(b)))
    for i in range(segments):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            c, d = ((i + 1) % segments) * sides + (j + 1) % sides, ((i + 1) % segments) * sides + j
            faces.append((a, b, c, d))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    return mesh


def disc_mesh(name, radius, segments=64):
    """Flat disc in the XZ plane, facing -Y."""
    verts = [(0, 0, 0)] + [(radius * math.cos(2 * math.pi * i / segments), 0,
                            radius * math.sin(2 * math.pi * i / segments)) for i in range(segments)]
    faces = [(0, i + 1, (i + 1) % segments + 1) for i in range(segments)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    return mesh


def wireframe(obj, thickness):
    mod = obj.modifiers.new("wire", "WIREFRAME")
    mod.thickness = thickness
    mod.use_replace = True
    return obj


# ── depth layers ─────────────────────────────────────────────────────────────

class FloorGrid:
    """Wire grid floor: gives a horizon line and perspective lines for free.
    Brightens slightly on transients."""

    def __init__(self, scn, mat, sig, size=80.0, cells=40, z=0.0, center=(0, 0), thickness=0.025):
        bm = bmesh.new()
        bmesh.ops.create_grid(bm, x_segments=cells, y_segments=cells, size=size / 2)
        self.obj = wireframe(new_obj(scn, "floor", _mesh_from_bmesh("floor", bm), mat,
                                     loc=(center[0], center[1], z)), thickness)
        self.sig = sig

    def update(self, f, base=0.08, gain=0.25):
        set_drive(self.obj, base + gain * self.sig.onset[f])


class Halo:
    """Large thin ring behind or around the subject: frames it and marks scale."""

    def __init__(self, scn, mat, sig, radius, tube=0.06, loc=(0, 0, 0), rot_deg=(0, 0, 0), accent=1.0):
        self.obj = new_obj(scn, "halo", ring_mesh("halo", radius, tube, 96, 6), mat, loc=loc, accent=accent)
        self.obj.rotation_euler = tuple(math.radians(a) for a in rot_deg)
        self.sig, self.accent = sig, accent

    def update(self, f, base=0.12, gain=0.7, signal=None):
        s = (signal or self.sig.beat)[f]
        set_drive(self.obj, base + gain * s, self.accent)


class Dust:
    """Sparse tiny particles on a distant shell: depth cue plus hi-hat sparkle."""

    def __init__(self, scn, mat, sig, count=140, r_min=20.0, r_max=45.0, z_min=-5.0, z_max=25.0,
                 center=(0, 0, 0), seed=3):
        self.sig = sig
        self.pivot = new_empty(scn, "dust_pivot", center)
        mesh = ico_mesh("dust", 0.07, 0)
        self.objs = []
        for i in range(count):
            a = 2 * math.pi * hash01(seed, i, 1)
            r = r_min + (r_max - r_min) * hash01(seed, i, 2)
            z = z_min + (z_max - z_min) * hash01(seed, i, 3)
            s = 0.6 + 1.8 * hash01(seed, i, 4)
            o = new_obj(scn, f"dust_{i}", mesh, mat, loc=(r * math.cos(a), r * math.sin(a), z), parent=self.pivot)
            o.scale = (s, s, s)
            self.objs.append((o, i))
        self.spin = sig.cumulative(lambda f: 0.01 + 0.03 * sig.energy[f])

    def update(self, f):
        sig = self.sig
        b = sig.beat_index[f]
        for o, i in self.objs:
            flicker = sig.high[f] * (1.0 if hash01(i, b, 7) > 0.6 else 0.25)
            set_drive(o, 0.05 + 0.8 * flicker)
        self.pivot.rotation_euler.z = self.spin[f]


# ── camera ───────────────────────────────────────────────────────────────────

class Shot:
    """pos(f, u) / target(f, u) return (x, y, z); u = seconds since this shot's
    cut, for slow push-ins and drifts within a shot. shift = lens shift for
    off-centre framing without moving the target; roll in degrees."""

    def __init__(self, lens, pos, target, shift=(0.0, 0.0), roll=0.0, shake=1.0):
        self.lens, self.pos, self.target = lens, pos, target
        self.shift, self.roll, self.shake = shift, roll, shake


class CameraRig:
    """Hard cut to the next shot at every sig.cut change (section changes and
    long sections); small shake on bass transients."""

    def __init__(self, scn, sig, shots, clip_end=400.0):
        self.sig, self.shots = sig, shots
        data = bpy.data.cameras.new("Cam")
        data.clip_end = clip_end
        data.sensor_width = 36.0
        self.cam = bpy.data.objects.new("Cam", data)
        self.cam.rotation_mode = "QUATERNION"
        scn.collection.objects.link(self.cam)
        scn.camera = self.cam

    def shot_at(self, f):
        return self.shots[self.sig.cut[f] % len(self.shots)]

    def update(self, f):
        sig = self.sig
        shot = self.shot_at(f)
        u = sig.since_cut[f]
        pos = Vector(shot.pos(f, u))
        target = Vector(shot.target(f, u))
        amp = 0.06 * shot.shake * (0.5 * sig.low[f] * sig.beat[f] + 0.5 * sig.onset[f])
        t = sig.t[f]
        pos += Vector((math.sin(t * 37.0), math.sin(t * 29.0 + 1.3), math.sin(t * 41.0 + 2.1))) * amp
        direction = target - pos
        if direction.length < 1e-6:
            direction = Vector((0, 1, 0))
        q = direction.to_track_quat("-Z", "Y") @ Quaternion((0, 0, 1), math.radians(shot.roll))
        self.cam.location = pos
        self.cam.rotation_quaternion = q
        self.cam.data.lens = shot.lens
        self.cam.data.shift_x, self.cam.data.shift_y = shot.shift


def orbit(radius, z, angle0=0.0, speed=0.05, center=(0.0, 0.0)):
    """pos fn: slow orbit starting at angle0 (radians), speed in rad/s."""
    return lambda f, u: (center[0] + radius * math.cos(angle0 + speed * u),
                         center[1] + radius * math.sin(angle0 + speed * u), z)


def fixed(x, y, z, drift=(0.0, 0.0, 0.0)):
    """pos/target fn: fixed point with an optional linear drift per second."""
    return lambda f, u: (x + drift[0] * u, y + drift[1] * u, z + drift[2] * u)


def tilt_matrix(rx, ry, rz):
    return (Matrix.Rotation(math.radians(rz), 3, "Z") @ Matrix.Rotation(math.radians(ry), 3, "Y")
            @ Matrix.Rotation(math.radians(rx), 3, "X"))
