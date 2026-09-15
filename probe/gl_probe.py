"""Headless OpenGL (moderngl) render-speed probe.

Usage: python gl_probe.py <out.mp4> <frames> <width> <height> <fps>

Creates a GPU OpenGL context without a display (EGL on Linux, trying several
library/device combinations and reporting which one gives a real GPU renderer
rather than llvmpipe), renders an audio-visualizer-like test scene -- ~400
instanced glowing primitives over a procedural grid floor -- into an HDR target
with a full post chain (bloom mip chain, light shafts, depth of field, ACES
tonemap, vignette, film grain), reads each frame back and pipes it to ffmpeg.
The clock is stepped manually (t = f / fps), so output duration is independent
of wall time. Prints PROBE_GL lines with per-stage timing as it goes.
"""
import math
import os
import struct
import subprocess
import sys
import time

import numpy as np


def log(line):
    print(f"PROBE_GL {line}", flush=True)


NVIDIA_EGL_VENDOR_FILES = ["/usr/share/glvnd/egl_vendor.d/10_nvidia.json", "/etc/glvnd/egl_vendor.d/10_nvidia.json",
                            "/usr/local/share/glvnd/egl_vendor.d/10_nvidia.json"]


def gpu_driver_diagnostics():
    """What the container exposes. Graydient hosts can have an AMD iGPU (Mesa)
    next to the NVIDIA card, and EGL/Vulkan may default to the iGPU."""
    import glob
    for pattern in ("/usr/share/glvnd/egl_vendor.d/*", "/etc/glvnd/egl_vendor.d/*", "/usr/share/vulkan/icd.d/*",
                    "/etc/vulkan/icd.d/*", "/dev/dri/*"):
        log(f"diag {pattern}: {sorted(glob.glob(pattern)) or 'none'}")
    for lib in ("libEGL_nvidia.so.0", "libEGL_mesa.so.0", "libEGL.so.1", "libOpenGL.so.0", "libGLX_nvidia.so.0"):
        found = subprocess.run(f"ldconfig -p | grep -F {lib}", shell=True, capture_output=True, text=True).stdout.strip()
        log(f"diag {lib}: {'present' if found else 'missing'}")


def ensure_moderngl():
    try:
        import moderngl  # noqa: F401
        return
    except ImportError:
        pass
    # Graydient dropped a pinned "moderngl==5.12.0" from the pip requirements
    # without an error, so install it here if it's missing.
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "moderngl"], capture_output=True, text=True)
    log(f"moderngl was missing; pip install rc={r.returncode} in {time.time() - t0:.1f}s"
        + ("" if r.returncode == 0 else f" err={r.stderr[-300:]}"))


def create_context():
    """Tries each EGL device and prefers a renderer that reports NVIDIA; falls
    back to any non-software renderer (e.g. an iGPU) so a result still exists."""
    import moderngl
    if sys.platform.startswith("linux") and "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ:
        vendor = next((p for p in NVIDIA_EGL_VENDOR_FILES if os.path.isfile(p)), None)
        if vendor:
            # glvnd reads this when libEGL loads, i.e. at the first context below.
            os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = vendor
            log(f"forcing NVIDIA EGL vendor: {vendor}")
    attempts = [{"backend": "egl", "device_index": i} for i in range(6)] + [{"backend": "egl"}] \
        if sys.platform.startswith("linux") else [{}]
    candidates, last = [], None
    for args in attempts:
        try:
            ctx = moderngl.create_context(standalone=True, require=330, **args)
        except Exception as e:  # noqa: BLE001 -- report every failure mode
            last = e
            log(f"context attempt {args} failed: {str(e).strip()[:160]}")
            continue
        renderer = ctx.info.get("GL_RENDERER", "?")
        software = any(s in renderer.lower() for s in ("llvmpipe", "softpipe", "swiftshader", "software"))
        nvidia = "nvidia" in renderer.lower()
        log(f"context attempt {args} -> renderer={renderer!r} version={ctx.info.get('GL_VERSION')} "
            f"nvidia={nvidia} software={software}")
        if nvidia:
            for c, _, _ in candidates:
                c.release()
            return ctx, args, renderer
        if software:
            ctx.release()
        else:
            candidates.append((ctx, args, renderer))
    if candidates:
        for c, _, _ in candidates[1:]:
            c.release()
        log(f"WARNING no NVIDIA renderer found, using {candidates[0][2]!r}")
        return candidates[0]
    raise RuntimeError(f"no GPU OpenGL context available (last error: {last})")


# ── geometry ─────────────────────────────────────────────────────────────────

def icosphere(subdiv=2):
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
             (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
             (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(v, dtype=np.float32) / np.linalg.norm(v) for v in verts]
    for _ in range(subdiv):
        cache, new_faces = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    tri = np.array([verts[i] for f in faces for i in f], dtype=np.float32)
    # flat normals for a faceted, readable CG look
    n = np.cross(tri[1::3] - tri[0::3], tri[2::3] - tri[0::3])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    normals = np.repeat(n, 3, axis=0)
    return np.hstack([tri, normals]).astype("f4")


def perspective(fovy, aspect, near, far):
    f = 1 / math.tan(fovy / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]], dtype="f4")


def look_at(eye, target, up=(0, 0, 1)):
    eye, target, up = (np.array(v, dtype="f4") for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4, dtype="f4")
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


# ── shaders ──────────────────────────────────────────────────────────────────

SCENE_VS = """
#version 330
uniform mat4 u_viewproj;
in vec3 in_pos; in vec3 in_norm;
in vec4 in_inst;   // xyz offset, w scale
in vec2 in_style;  // drive, accent
out vec3 v_norm; out vec3 v_world; out vec2 v_style;
void main() {
    vec3 world = in_pos * in_inst.w + in_inst.xyz;
    v_norm = in_norm; v_world = world; v_style = in_style;
    gl_Position = u_viewproj * vec4(world, 1.0);
}
"""

SCENE_FS = """
#version 330
uniform vec3 u_eye;
in vec3 v_norm; in vec3 v_world; in vec2 v_style;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_depth;
vec3 ramp(float d) {
    vec3 a = vec3(0.01, 0.015, 0.2), b = vec3(0.03, 0.25, 1.0), c = vec3(1.0, 0.3, 0.04);
    return d < 0.5 ? mix(a, b, d * 2.0) : mix(b, c, d * 2.0 - 1.0);
}
void main() {
    float drive = v_style.x;
    vec3 base = mix(ramp(drive), vec3(1.0, 0.04, 0.45), v_style.y);
    vec3 n = normalize(v_norm);
    vec3 v = normalize(u_eye - v_world);
    float key = max(dot(n, normalize(vec3(0.4, -0.5, 0.8))), 0.0);
    float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    vec3 col = base * (0.08 + 0.6 * key) + vec3(1.0, 0.25, 0.6) * rim * 0.8 + base * drive * drive * 6.0;
    out_color = vec4(col, 1.0);
    out_depth = vec4(length(u_eye - v_world), 0.0, 0.0, 1.0);
}
"""

FLOOR_VS = """
#version 330
uniform mat4 u_viewproj;
in vec2 in_xy;
out vec3 v_world;
void main() { v_world = vec3(in_xy * 80.0, -2.2); gl_Position = u_viewproj * vec4(v_world, 1.0); }
"""

FLOOR_FS = """
#version 330
uniform vec3 u_eye; uniform float u_pulse;
in vec3 v_world;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_depth;
void main() {
    vec2 g = abs(fract(v_world.xy / 2.0 - 0.5) - 0.5) / fwidth(v_world.xy / 2.0);
    float line = 1.0 - min(min(g.x, g.y), 1.0);
    float dist = length(u_eye - v_world);
    float fade = exp(-dist * 0.025);
    vec3 col = vec3(0.05, 0.2, 1.0) * line * (0.6 + 2.5 * u_pulse) * fade + vec3(0.002, 0.002, 0.008);
    out_color = vec4(col, 1.0);
    out_depth = vec4(dist, 0.0, 0.0, 1.0);
}
"""

QUAD_VS = """
#version 330
in vec2 in_xy; out vec2 uv;
void main() { uv = in_xy * 0.5 + 0.5; gl_Position = vec4(in_xy, 0.0, 1.0); }
"""

BRIGHT_FS = """
#version 330
uniform sampler2D src; in vec2 uv; out vec4 o;
void main() { vec3 c = texture(src, uv).rgb; float l = max(max(c.r, c.g), c.b);
              o = vec4(c * smoothstep(0.9, 1.6, l), 1.0); }
"""

DOWN_FS = """
#version 330
uniform sampler2D src; uniform vec2 texel; in vec2 uv; out vec4 o;
void main() {   // dual-filter downsample
    vec3 c = texture(src, uv).rgb * 4.0;
    c += texture(src, uv + texel * vec2(-1, -1)).rgb + texture(src, uv + texel * vec2(1, -1)).rgb;
    c += texture(src, uv + texel * vec2(-1, 1)).rgb + texture(src, uv + texel * vec2(1, 1)).rgb;
    o = vec4(c / 8.0, 1.0);
}
"""

UP_FS = """
#version 330
uniform sampler2D src; uniform sampler2D prev; uniform vec2 texel; in vec2 uv; out vec4 o;
void main() {   // tent upsample + add the next-finer level
    vec3 c = vec3(0.0);
    c += texture(src, uv + texel * vec2(-1, 0)).rgb * 2.0 + texture(src, uv + texel * vec2(1, 0)).rgb * 2.0;
    c += texture(src, uv + texel * vec2(0, -1)).rgb * 2.0 + texture(src, uv + texel * vec2(0, 1)).rgb * 2.0;
    c += texture(src, uv + texel * vec2(-1, -1)).rgb + texture(src, uv + texel * vec2(1, -1)).rgb;
    c += texture(src, uv + texel * vec2(-1, 1)).rgb + texture(src, uv + texel * vec2(1, 1)).rgb;
    o = vec4(c / 12.0 + texture(prev, uv).rgb, 1.0);
}
"""

RAYS_FS = """
#version 330
uniform sampler2D src; uniform vec2 light_uv; in vec2 uv; out vec4 o;
void main() {   // screen-space light shafts from the bright pass
    vec2 d = (uv - light_uv) / 40.0; vec2 p = uv; vec3 acc = vec3(0.0); float w = 1.0;
    for (int i = 0; i < 40; i++) { p -= d; acc += texture(src, p).rgb * w; w *= 0.96; }
    o = vec4(acc / 40.0, 1.0);
}
"""

COMPOSITE_FS = """
#version 330
uniform sampler2D color; uniform sampler2D depth; uniform sampler2D blurred; uniform sampler2D bloom;
uniform sampler2D rays; uniform float focus; uniform float frame; in vec2 uv; out vec4 o;
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233)) + frame * 0.618) * 43758.5453); }
void main() {
    float d = texture(depth, uv).r;
    float coc = clamp(abs(d - focus) / (d * 0.6 + 0.001) * 1.4, 0.0, 1.0);    // shallow depth of field
    vec3 c = mix(texture(color, uv).rgb, texture(blurred, uv).rgb, coc);
    c += texture(bloom, uv).rgb * 0.9 + texture(rays, uv).rgb * 0.6;
    c = aces(c * 0.9);
    vec2 q = uv - 0.5; c *= 1.0 - dot(q, q) * 1.1;                          // vignette
    c += (hash(uv * 1000.0) - 0.5) * 0.045;                                   // film grain
    o = vec4(pow(max(c, 0.0), vec3(1.0 / 2.2)), 1.0);
}
"""


def main():
    out_path, frames, width, height, fps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    if sys.platform.startswith("linux"):
        gpu_driver_diagnostics()
    ensure_moderngl()
    t0 = time.time()
    ctx, ctx_args, renderer = create_context()
    log(f"GPU context in {time.time() - t0:.2f}s via {ctx_args}: {renderer}")
    import moderngl

    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4"))

    def prog(vs, fs):
        return ctx.program(vertex_shader=vs, fragment_shader=fs)

    scene_p, floor_p = prog(SCENE_VS, SCENE_FS), prog(FLOOR_VS, FLOOR_FS)
    bright_p, down_p, up_p = prog(QUAD_VS, BRIGHT_FS), prog(QUAD_VS, DOWN_FS), prog(QUAD_VS, UP_FS)
    rays_p, comp_p = prog(QUAD_VS, RAYS_FS), prog(QUAD_VS, COMPOSITE_FS)

    def quad_vao(p):
        return ctx.vertex_array(p, [(quad, "2f", "in_xy")])

    mesh = ctx.buffer(icosphere(2))
    grid_n = 20
    n_inst = grid_n * grid_n
    inst_buf = ctx.buffer(reserve=n_inst * 16)
    style_buf = ctx.buffer(reserve=n_inst * 8)
    scene_vao = ctx.vertex_array(scene_p, [(mesh, "3f 3f", "in_pos", "in_norm"),
                                           (inst_buf, "4f/i", "in_inst"), (style_buf, "2f/i", "in_style")])
    floor_vao = ctx.vertex_array(floor_p, [(quad, "2f", "in_xy")])

    def tex(w, h, comps=4, dtype="f2"):
        t = ctx.texture((w, h), comps, dtype=dtype)
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = t.repeat_y = False
        return t

    color_t, depth_t = tex(width, height), tex(width, height, 4, "f4")
    scene_fbo = ctx.framebuffer([color_t, depth_t], ctx.depth_renderbuffer((width, height)))
    half = (width // 2, height // 2)
    bright_t = tex(*half)
    bright_fbo = ctx.framebuffer([bright_t])
    levels = []
    w, h = half
    for _ in range(5):
        w, h = max(w // 2, 1), max(h // 2, 1)
        t = tex(w, h)
        levels.append((t, ctx.framebuffer([t])))
    ups = [(tex(t.width, t.height), None) for t, _ in levels[:-1]]
    ups = [(t, ctx.framebuffer([t])) for t, _ in ups]
    rays_t = tex(*half)
    rays_fbo = ctx.framebuffer([rays_t])
    out_t = ctx.texture((width, height), 3, dtype="f1")
    out_fbo = ctx.framebuffer([out_t])
    bright_vao, down_vao, up_vao = quad_vao(bright_p), quad_vao(down_p), quad_vao(up_p)
    rays_vao, comp_vao = quad_vao(rays_p), quad_vao(comp_p)

    probe = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=320x240:rate=24:duration=0.2",
                            "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True)
    enc = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "19", "-b:v", "0"] if probe.returncode == 0 \
        else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    log(f"encoder: {enc[1]}")
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                           "-framerate", str(fps), "-i", "-", "-vf", "vflip", *enc, "-pix_fmt", "yuv420p", out_path],
                          stdin=subprocess.PIPE)

    proj = perspective(math.radians(42), width / height, 0.1, 300.0)
    half_n = (grid_n - 1) / 2
    gx, gy = np.meshgrid((np.arange(grid_n) - half_n) * 0.55, (np.arange(grid_n) - half_n) * 0.55)
    gx, gy = gx.ravel().astype("f4"), gy.ravel().astype("f4")
    dist = np.sqrt(gx * gx + gy * gy)
    inst = np.zeros((n_inst, 4), dtype="f4")
    style = np.zeros((n_inst, 2), dtype="f4")

    t_render = t_read = t_write = 0.0
    t_start = time.time()
    for f in range(frames):
        t = f / fps
        beat = math.exp(-((t * 128 / 60) % 1.0) * 60 / 128 / 0.15)
        amp = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(t * 0.9))
        pulse_r = ((t * 128 / 60) % 1.0) * 60 / 128 * 7.0
        ring = np.exp(-((dist - pulse_r) / 0.7) ** 2)
        z = amp * np.sin(2 * math.pi * (t * 0.6 - dist / 6.0)) + 0.9 * beat * ring
        drive = np.clip(0.5 + 0.5 * z / (amp + 0.9), 0, 1)
        inst[:, 0], inst[:, 1], inst[:, 2], inst[:, 3] = gx, gy, z, 0.16 * (0.6 + 0.6 * drive)
        style[:, 0], style[:, 1] = drive, np.clip(ring * beat * 1.5, 0, 1)
        a = -0.9 + t * 0.12
        eye = (10 * math.cos(a), 10 * math.sin(a), 2.2 + math.sin(t * 0.3))
        view_proj = (proj @ look_at(eye, (0, 0, -0.4))).T.copy()

        r0 = time.time()
        inst_buf.write(inst.tobytes())
        style_buf.write(style.tobytes())
        scene_fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        for p in (scene_p, floor_p):
            p["u_viewproj"].write(view_proj.tobytes())
            p["u_eye"].value = eye
        floor_p["u_pulse"].value = beat
        floor_vao.render(moderngl.TRIANGLE_STRIP)
        scene_vao.render(moderngl.TRIANGLES, instances=n_inst)
        ctx.disable(moderngl.DEPTH_TEST)

        bright_fbo.use()
        color_t.use(0)
        bright_p["src"].value = 0
        bright_vao.render(moderngl.TRIANGLE_STRIP)
        src = bright_t
        for t_lvl, fbo in levels:
            fbo.use()
            src.use(0)
            down_p["src"].value = 0
            down_p["texel"].value = (1 / src.width, 1 / src.height)
            down_vao.render(moderngl.TRIANGLE_STRIP)
            src = t_lvl
        for i in range(len(levels) - 2, -1, -1):
            t_up, fbo = ups[i]
            fbo.use()
            src.use(0)
            levels[i][0].use(1)
            up_p["src"].value, up_p["prev"].value = 0, 1
            up_p["texel"].value = (1 / src.width, 1 / src.height)
            up_vao.render(moderngl.TRIANGLE_STRIP)
            src = t_up
        bloom_t = src

        center = view_proj.T @ np.array([0, 0, 0.5, 1], dtype="f4")
        light_uv = (center[0] / center[3] * 0.5 + 0.5, center[1] / center[3] * 0.5 + 0.5)
        rays_fbo.use()
        bright_t.use(0)
        rays_p["src"].value = 0
        rays_p["light_uv"].value = light_uv
        rays_vao.render(moderngl.TRIANGLE_STRIP)

        out_fbo.use()
        color_t.use(0)
        depth_t.use(1)
        levels[0][0].use(2)
        bloom_t.use(3)
        rays_t.use(4)
        for name, unit in (("color", 0), ("depth", 1), ("blurred", 2), ("bloom", 3), ("rays", 4)):
            comp_p[name].value = unit
        comp_p["focus"].value = 10.0
        comp_p["frame"].value = float(f)
        comp_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.finish()
        r1 = time.time()
        data = out_fbo.read(components=3, alignment=1)
        r2 = time.time()
        ff.stdin.write(data)
        r3 = time.time()
        if f >= 10:  # skip warm-up (shader compile, first allocations)
            t_render, t_read, t_write = t_render + r1 - r0, t_read + r2 - r1, t_write + r3 - r2
        if f in (0, 9) or (f + 1) % max(1, frames // 5) == 0:
            log(f"frame {f + 1}/{frames} elapsed {time.time() - t_start:.1f}s")
    ff.stdin.close()
    rc = ff.wait()
    n = max(frames - 10, 1)
    wall = time.time() - t_start
    log(f"RESULT {width}x{height} {frames} frames: gpu_render+post={t_render / n * 1000:.2f}ms "
        f"readback={t_read / n * 1000:.2f}ms encode_write={t_write / n * 1000:.2f}ms "
        f"wall={wall:.1f}s ({wall / frames * 1000:.2f}ms/frame, {frames / wall:.1f} fps, "
        f"{frames / fps / wall:.1f}x realtime) ffmpeg_rc={rc}")
    sys.exit(0 if rc == 0 else 3)


if __name__ == "__main__":
    main()
