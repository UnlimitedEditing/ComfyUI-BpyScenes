"""GL core for the music visualizer: context, meshes, instanced batches, camera.

Everything per-frame is numpy-vectorised: a batch's instances are updated with a
handful of array operations, never a Python loop over elements.
"""
import math
import os
import sys

import numpy as np

NVIDIA_EGL_VENDOR_FILES = ["/usr/share/glvnd/egl_vendor.d/10_nvidia.json", "/etc/glvnd/egl_vendor.d/10_nvidia.json",
                            "/usr/local/share/glvnd/egl_vendor.d/10_nvidia.json"]


def create_context(log):
    """GPU OpenGL 3.3 context without a display. Graydient containers ship both
    the NVIDIA and Mesa EGL vendors and some hosts have an AMD iGPU, so the
    NVIDIA vendor is forced and an NVIDIA renderer preferred (probe-verified:
    EGL device 0 with the vendor forced -> RTX 4090)."""
    import moderngl
    linux = sys.platform.startswith("linux")
    if linux and "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ:
        vendor = next((p for p in NVIDIA_EGL_VENDOR_FILES if os.path.isfile(p)), None)
        if vendor:
            os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = vendor
    attempts = ([{"backend": "egl", "device_index": i} for i in range(6)] + [{"backend": "egl"}]) if linux else [{}]
    fallback, last = None, None
    for args in attempts:
        try:
            ctx = moderngl.create_context(standalone=True, require=330, **args)
        except Exception as e:  # noqa: BLE001
            last = e
            continue
        renderer = ctx.info.get("GL_RENDERER", "?")
        if "nvidia" in renderer.lower():
            if fallback:
                fallback[0].release()
            log(f"GL context: {renderer} via {args}")
            return ctx
        if any(s in renderer.lower() for s in ("llvmpipe", "softpipe", "software")) or fallback:
            ctx.release()
        else:
            fallback = (ctx, renderer)
    if fallback:
        log(f"WARNING GL context: no NVIDIA renderer, using {fallback[1]}")
        return fallback[0]
    raise RuntimeError(f"no GPU OpenGL context available: {last}")


# ── meshes: (N*3, 6) float32 arrays of position + flat normal ────────────────

def _flat(tris):
    tris = np.asarray(tris, dtype=np.float32).reshape(-1, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return np.concatenate([tris.reshape(-1, 3), np.repeat(n, 3, axis=0)], axis=1).astype("f4")


def ico_mesh(radius=1.0, subdiv=1):
    t = (1 + 5 ** 0.5) / 2
    verts = [np.array(v, dtype=np.float64) for v in
             [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
              (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]]
    verts = [v / np.linalg.norm(v) for v in verts]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
             (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
             (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(subdiv):
        cache, new = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    v = np.array(verts) * radius
    return _flat(v[np.array(faces)]), v, faces


def box_mesh(w, d, h, base=True):
    z0 = 0.0 if base else -h / 2
    c = np.array([(-w / 2, -d / 2, z0), (w / 2, -d / 2, z0), (w / 2, d / 2, z0), (-w / 2, d / 2, z0),
                  (-w / 2, -d / 2, z0 + h), (w / 2, -d / 2, z0 + h), (w / 2, d / 2, z0 + h), (-w / 2, d / 2, z0 + h)])
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return _flat([c[[a, b, cc, a, cc, d]] for a, b, cc, d in quads])


def ring_mesh(radius, tube, segments=48, sides=6):
    """Torus in the XY plane (axis Z)."""
    a = 2 * np.pi * np.arange(segments) / segments
    b = 2 * np.pi * np.arange(sides) / sides
    r = radius + tube * np.cos(b)[None, :]
    shape = (segments, sides)
    pts = np.stack([r * np.cos(a)[:, None], r * np.sin(a)[:, None], np.broadcast_to(tube * np.sin(b), shape)], -1)
    tris = []
    for i in range(segments):
        for j in range(sides):
            p00, p01 = pts[i, j], pts[i, (j + 1) % sides]
            p10, p11 = pts[(i + 1) % segments, j], pts[(i + 1) % segments, (j + 1) % sides]
            tris += [p00, p01, p11, p00, p11, p10]
    return _flat(tris)


def disc_mesh(radius, segments=64):
    """Flat disc in the XZ plane facing -Y."""
    a = 2 * np.pi * np.arange(segments + 1) / segments
    rim = np.stack([radius * np.cos(a), np.zeros_like(a), radius * np.sin(a)], -1)
    tris = []
    for i in range(segments):
        tris += [(0, 0, 0), rim[i + 1], rim[i]]
    return _flat(tris)


def wire_mesh(verts, faces, thickness):
    """Edges of a mesh as thin square struts -- the GL equivalent of Blender's
    wireframe modifier."""
    edges = {tuple(sorted((f[i], f[(i + 1) % 3]))) for f in faces for i in range(3)}
    tris = []
    for a, b in edges:
        p, q = verts[a], verts[b]
        axis = (q - p) / np.linalg.norm(q - p)
        u = np.cross(axis, [0, 0, 1] if abs(axis[2]) < 0.9 else [1, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        h = thickness / 2
        ring = [u * h + v * h, -u * h + v * h, -u * h - v * h, u * h - v * h]
        for k in range(4):
            c0, c1 = ring[k], ring[(k + 1) % 4]
            tris += [p + c0, q + c0, q + c1, p + c0, q + c1, p + c1]
    return _flat(tris)


# ── instancing ───────────────────────────────────────────────────────────────

def model_matrices(pos, scale=None, rot=None):
    """Column-major (n, 16) float32 model matrices. rot is Blender-style XYZ euler
    (applied X, then Y, then Z) so positions/rotations port 1:1 from bpy scenes."""
    pos = np.asarray(pos, dtype=np.float32)
    n = len(pos)
    m = np.zeros((n, 4, 4), dtype=np.float32)
    if rot is None:
        r = np.broadcast_to(np.eye(3, dtype=np.float32), (n, 3, 3)).copy()
    else:
        rot = np.broadcast_to(np.asarray(rot, dtype=np.float32), (n, 3))
        cx, cy, cz = np.cos(rot).T
        sx, sy, sz = np.sin(rot).T
        r = np.empty((n, 3, 3), dtype=np.float32)
        r[:, 0, 0], r[:, 0, 1], r[:, 0, 2] = cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz
        r[:, 1, 0], r[:, 1, 1], r[:, 1, 2] = cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz
        r[:, 2, 0], r[:, 2, 1], r[:, 2, 2] = -sy, sx * cy, cx * cy
    if scale is not None:
        r *= np.broadcast_to(np.asarray(scale, dtype=np.float32), (n, 3))[:, None, :]
    m[:, :3, :3] = r
    m[:, :3, 3] = pos
    m[:, 3, 3] = 1.0
    return m.transpose(0, 2, 1).reshape(n, 16)


class Batch:
    """One mesh drawn n times. Set .pos/.scale/.rot/.drive/.accent (numpy arrays
    or scalars) each frame; upload() packs them into the instance buffer."""

    def __init__(self, ctx, program, mesh, count):
        self.count = count
        self.vbo = ctx.buffer(mesh.tobytes())
        self.ibo = ctx.buffer(reserve=count * 18 * 4)
        self.vao = ctx.vertex_array(program, [(self.vbo, "3f 3f", "in_pos", "in_norm"),
                                              (self.ibo, "16f 2f/i", "in_model", "in_style")])
        self.pos = np.zeros((count, 3), dtype=np.float32)
        self.scale = np.ones((count, 3), dtype=np.float32)
        self.rot = None
        self.drive = np.zeros(count, dtype=np.float32)
        self.accent = np.zeros(count, dtype=np.float32)
        self.visible = True

    def upload(self):
        data = np.empty((self.count, 18), dtype=np.float32)
        data[:, :16] = model_matrices(self.pos, self.scale, self.rot)
        data[:, 16] = np.clip(np.broadcast_to(self.drive, self.count), 0, 1)
        data[:, 17] = np.clip(np.broadcast_to(self.accent, self.count), 0, 1)
        self.ibo.write(data.tobytes())

    def render(self, mode):
        if self.visible:
            self.vao.render(mode, instances=self.count)


# ── camera ───────────────────────────────────────────────────────────────────

def view_matrix(eye, target, roll_deg=0.0):
    eye, target = np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64)
    f = target - eye
    f /= max(np.linalg.norm(f), 1e-9)
    up = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.999 else np.array([0.0, 1.0, 0.0])
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    roll = math.radians(roll_deg)
    s, u = s * math.cos(roll) + u * math.sin(roll), -s * math.sin(roll) + u * math.cos(roll)
    m = np.identity(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def projection_matrix(lens_mm, aspect, shift=(0.0, 0.0), near=0.1, far=500.0, sensor_mm=36.0):
    """Blender-style: horizontal sensor fit, lens shift as a fraction of the
    larger image dimension."""
    fx = 2 * lens_mm / sensor_mm
    fy = fx * aspect
    big = max(1.0, aspect)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = fx, fy
    m[0, 2] = -2 * shift[0] * big
    m[1, 2] = -2 * shift[1] * big / aspect
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def euler_dir(rx, ry, rz, v=(0.0, 0.0, -1.0)):
    """Direction v rotated by a Blender XYZ euler (degrees) -- e.g. a sun's travel direction."""
    r = model_matrices([[0, 0, 0]], rot=[[math.radians(rx), math.radians(ry), math.radians(rz)]])[0].reshape(4, 4).T
    return r[:3, :3] @ np.asarray(v, dtype=np.float32)


def hash01(*keys):
    """Vectorised deterministic 0-1 hash of integer keys (broadcasts)."""
    arrays = np.broadcast_arrays(*[np.asarray(k, dtype=np.int64) for k in keys])
    h = np.full(arrays[0].shape, 2166136261, dtype=np.uint64)
    for k in arrays:
        h = ((h ^ (k & 0xFFFFFFFF).astype(np.uint64)) * np.uint64(16777619)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(13)
    h = (h * np.uint64(1274126177)) & np.uint64(0xFFFFFFFF)
    return (h & np.uint64(0xFFFFFF)).astype(np.float64) / float(0x1000000)
