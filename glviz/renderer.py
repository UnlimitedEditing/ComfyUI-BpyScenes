"""Scene + post renderer. Pass order per frame:

  sky gradient -> instanced batches (lit, fogged, HDR emission) -> additive grid
  floor -> bright pass -> 6-level dual-filter bloom -> light shafts from the
  scene's hero point -> blurred colour for depth of field -> composite (DoF,
  bloom, shafts, chromatic aberration, ACES, grade, vignette, grain) at the
  output size (downsampling from the render size when supersampling).

Looks set every colour; scenes only provide geometry, drive/accent, camera and
floor/hero parameters (see scenes.py).
"""
import math

import moderngl
import numpy as np

from gpu import Batch, euler_dir, projection_matrix, view_matrix

SCENE_VS = """
#version 330
uniform mat4 u_viewproj;
in vec3 in_pos; in vec3 in_norm; in mat4 in_model; in vec2 in_style;
out vec3 v_world; out vec3 v_norm; out vec2 v_style;
void main() {
    vec4 w = in_model * vec4(in_pos, 1.0);
    v_world = w.xyz; v_norm = mat3(in_model) * in_norm; v_style = in_style;
    gl_Position = u_viewproj * w;
}
"""

LOOK_GLSL = """
uniform vec3 u_ramp0; uniform vec3 u_ramp1; uniform vec3 u_ramp2; uniform float u_ramp_mid;
uniform vec3 u_accent; uniform vec2 u_strength; uniform vec3 u_horizon; uniform vec3 u_fog;
vec3 ramp(float d) {
    return d < u_ramp_mid ? mix(u_ramp0, u_ramp1, d / u_ramp_mid)
                          : mix(u_ramp1, u_ramp2, (d - u_ramp_mid) / (1.0 - u_ramp_mid));
}
vec3 palette(float drive, float accent) {
    return mix(ramp(drive), u_accent * mix(0.25, 1.0, drive), accent);
}
float fog_amount(float dist) { return clamp((dist - u_fog.x) / (u_fog.y - u_fog.x), 0.0, 1.0) * u_fog.z; }
"""

SCENE_FS = """
#version 330
uniform vec3 u_eye; uniform vec3 u_key_dir; uniform vec3 u_key_col; uniform vec3 u_rim_dir; uniform vec3 u_rim_col;
uniform float u_emit_gain;
""" + LOOK_GLSL + """
in vec3 v_world; in vec3 v_norm; in vec2 v_style;
layout(location = 0) out vec4 o_color; layout(location = 1) out vec4 o_depth;
void main() {
    float drive = v_style.x;
    vec3 base = palette(drive, v_style.y);
    vec3 nn = gl_FrontFacing ? v_norm : -v_norm;
    vec3 n = nn / max(length(nn), 1e-6);
    vec3 to_eye = u_eye - v_world;
    float dist = length(to_eye);
    vec3 v = to_eye / max(dist, 1e-6);
    // Keep pow() bases non-negative: with unit vectors dot() can land a hair
    // above 1.0, and pow(negative, y) is NaN -- a single NaN pixel gets
    // smeared by the bloom mip chain into a large black block on screen.
    float ndv = clamp(dot(n, v), 0.0, 1.0);
    // Surfaces carry the form (key diffuse + specular + rim light); emission is
    // reserved for excited elements -- drive^2 -- so calm geometry reads as
    // lit solids instead of flat glowing slabs. Strengths are the Blender look
    // values scaled for this HDR + bloom + ACES pipeline.
    vec3 surf = mix(base, vec3(dot(base, vec3(0.333))), 0.35) + 0.04;
    vec3 hk = u_key_dir + v;
    vec3 h = hk / max(length(hk), 1e-6);
    vec3 lit = surf * (0.03 + max(dot(n, u_key_dir), 0.0) * u_key_col * 0.9);
    lit += u_key_col * pow(clamp(dot(n, h), 0.0, 1.0), 48.0) * 0.35;
    lit += surf * max(dot(n, u_rim_dir), 0.0) * u_rim_col * 0.35;
    lit += u_rim_col * pow(1.0 - ndv, 4.0) * 0.1 * (0.3 + drive);
    vec3 emit = base * mix(u_strength.x, u_strength.y, drive * drive) * u_emit_gain;
    float fog = fog_amount(dist);
    o_color = vec4(min(mix(lit + emit * (1.0 - 0.7 * fog), u_horizon, fog), vec3(64.0)), 1.0);
    o_depth = vec4(dist, 0.0, 0.0, 1.0);
}
"""

FLOOR_VS = """
#version 330
uniform mat4 u_viewproj; uniform vec4 u_floor;  // center xy, z, half size
in vec2 in_xy; out vec3 v_world;
void main() {
    v_world = vec3(u_floor.xy + in_xy * u_floor.w, u_floor.z);
    gl_Position = u_viewproj * vec4(v_world, 1.0);
}
"""

FLOOR_FS = """
#version 330
uniform vec3 u_eye; uniform float u_floor_step; uniform float u_floor_drive; uniform float u_floor_width;
""" + LOOK_GLSL + """
in vec3 v_world; out vec4 o_color;
void main() {
    vec2 g = v_world.xy / u_floor_step;
    vec2 d = abs(fract(g - 0.5) - 0.5) / max(fwidth(g), vec2(1e-6));
    float line = 1.0 - clamp(min(d.x, d.y) / u_floor_width, 0.0, 1.0);
    float dist = length(u_eye - v_world);
    float fade = 1.0 - fog_amount(dist);
    vec3 c = ramp(u_floor_drive) * mix(0.3, u_strength.y, u_floor_drive) * line * fade;
    o_color = vec4(c, 1.0);
}
"""

SKY_FS = """
#version 330
uniform mat4 u_inv_viewproj; uniform vec3 u_eye; uniform vec3 u_zenith;
""" + LOOK_GLSL + """
in vec2 uv; layout(location = 0) out vec4 o_color; layout(location = 1) out vec4 o_depth;
void main() {
    vec4 p = u_inv_viewproj * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 dir = normalize(p.xyz / p.w - u_eye);
    float h = clamp((dir.z + 0.3) / 0.6, 0.0, 1.0);
    vec3 ground = u_zenith * 0.5;
    vec3 c = h < 0.47 ? mix(ground, u_horizon, h / 0.47)
           : h < 0.53 ? u_horizon : mix(u_horizon, u_zenith, clamp((h - 0.53) / 0.22, 0.0, 1.0));
    o_color = vec4(c, 1.0); o_depth = vec4(1000.0, 0.0, 0.0, 1.0);
}
"""

QUAD_VS = """
#version 330
in vec2 in_xy; out vec2 uv;
void main() { uv = in_xy * 0.5 + 0.5; gl_Position = vec4(in_xy, 0.0, 1.0); }
"""

BRIGHT_FS = """
#version 330
uniform sampler2D src; uniform float threshold; in vec2 uv; out vec4 o;
void main() {
    vec3 c = texture(src, uv).rgb;
    // Belt and braces: never let a non-finite pixel enter the bloom chain.
    if (any(isnan(c)) || any(isinf(c))) c = vec3(0.0);
    c = min(c, vec3(64.0));
    float l = max(max(c.r, c.g), c.b);
    o = vec4(c * smoothstep(threshold * 0.6, threshold * 1.6, l), 1.0);
}
"""

DOWN_FS = """
#version 330
uniform sampler2D src; uniform vec2 texel; in vec2 uv; out vec4 o;
void main() {
    vec3 c = texture(src, uv).rgb * 4.0;
    c += texture(src, uv + texel * vec2(-1, -1)).rgb + texture(src, uv + texel * vec2(1, -1)).rgb;
    c += texture(src, uv + texel * vec2(-1, 1)).rgb + texture(src, uv + texel * vec2(1, 1)).rgb;
    o = vec4(c / 8.0, 1.0);
}
"""

UP_FS = """
#version 330
uniform sampler2D src; uniform sampler2D prev; uniform vec2 texel; in vec2 uv; out vec4 o;
void main() {
    vec3 c = (texture(src, uv + texel * vec2(-1, 0)).rgb + texture(src, uv + texel * vec2(1, 0)).rgb
            + texture(src, uv + texel * vec2(0, -1)).rgb + texture(src, uv + texel * vec2(0, 1)).rgb) * 2.0;
    c += texture(src, uv + texel * vec2(-1, -1)).rgb + texture(src, uv + texel * vec2(1, -1)).rgb;
    c += texture(src, uv + texel * vec2(-1, 1)).rgb + texture(src, uv + texel * vec2(1, 1)).rgb;
    o = vec4(c / 12.0 + texture(prev, uv).rgb, 1.0);
}
"""

RAYS_FS = """
#version 330
uniform sampler2D src; uniform vec2 light_uv; uniform float amount; in vec2 uv; out vec4 o;
void main() {
    vec2 d = (uv - light_uv) / 48.0; vec2 p = uv; vec3 acc = vec3(0.0); float w = 1.0;
    for (int i = 0; i < 48; i++) { p -= d; acc += texture(src, p).rgb * w; w *= 0.955; }
    o = vec4(acc / 48.0 * amount, 1.0);
}
"""

COMPOSITE_FS = """
#version 330
uniform sampler2D color; uniform sampler2D depth; uniform sampler2D blur_half; uniform sampler2D blur_quarter;
uniform sampler2D bloom; uniform sampler2D rays;
uniform float focus; uniform float dof; uniform float bloom_amount; uniform float aberration; uniform float exposure;
uniform float grain; uniform float vignette; uniform vec3 grade; uniform float frame;
in vec2 uv; out vec4 o;
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233)) + frame * 0.61803) * 43758.5453); }
vec3 scene_at(vec2 st) {
    float d = texture(depth, st).r;
    float coc = clamp(abs(d - focus) / max(d, 0.001) * dof, 0.0, 1.0);
    vec3 sharp = texture(color, st).rgb;
    vec3 soft = mix(texture(blur_half, st).rgb, texture(blur_quarter, st).rgb, smoothstep(0.4, 1.0, coc));
    return mix(sharp, soft, smoothstep(0.05, 0.6, coc));
}
void main() {
    vec2 q = uv - 0.5;
    vec2 off = q * aberration;
    vec3 c = vec3(scene_at(uv + off).r, scene_at(uv).g, scene_at(uv - off).b);
    c += texture(bloom, uv).rgb * bloom_amount + texture(rays, uv).rgb;
    if (any(isnan(c)) || any(isinf(c))) c = texture(color, uv).rgb;
    if (any(isnan(c)) || any(isinf(c))) c = vec3(0.0);
    c = aces(c * grade.z * exposure);
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(l), c, grade.x);
    c = clamp((c - 0.5) * grade.y + 0.5, 0.0, 1.0);
    c *= 1.0 - dot(q, q) * vignette;
    c += (hash(uv * 1733.0) - 0.5) * grain;
    o = vec4(pow(max(c, 0.0), vec3(1.0 / 2.2)), 1.0);
}
"""


# ── user-facing post knobs ───────────────────────────────────────────────────
# Constrain-by-construction: every knob is 0-100 in the UI and maps through a
# curve onto an internal range tuned to look acceptable at both ends, so a bad
# value can't be expressed at all. 50 is the look's tuned baseline; defaults
# (in the node) sit a little brighter than that.

def _smooth(x):
    """S-curve on 0-1: gentle near both ends, responsive in the middle."""
    return x * x * (3.0 - 2.0 * x)


def _lerp(a, b, t):
    return a + (b - a) * t


def post_knob_values(knobs):
    """knobs: {"glow", "exposure", "light_rays", "depth_of_field"} each 0-100.
    Returns internal multipliers."""
    def k(name, default=50.0):
        return min(max(float(knobs.get(name, default)), 0.0), 100.0) / 100.0

    glow = _smooth(k("glow"))
    return {
        # glow drives emission and bloom together so they stay balanced:
        # 0 -> matte, lit solids; 100 -> hot neon (the early blown-out look is ~85)
        "emit": _lerp(0.2, 1.1, glow),
        "bloom": _lerp(0.15, 1.3, glow),
        "exposure": _lerp(0.55, 1.45, _smooth(k("exposure"))),
        "rays": _lerp(0.0, 1.6, k("light_rays")),
        "dof": _lerp(0.0, 2.0, k("depth_of_field")),
    }


class Renderer:
    def __init__(self, ctx, render_size, output_size, look, knobs=None):
        self.ctx, self.look = ctx, look
        self.rw, self.rh = render_size
        self.ow, self.oh = output_size
        self.batches = []
        # Scenes set these each frame.
        self.camera = {"eye": (10, 0, 2), "target": (0, 0, 0), "lens": 35.0, "shift": (0.0, 0.0), "roll": 0.0}
        self.floor = None      # {"center": (x, y), "z": z, "half": h, "step": s, "drive": d}
        self.hero = (0.0, 0.0, 0.0)
        self.clip_far = 500.0

        def prog(vs, fs):
            return ctx.program(vertex_shader=vs, fragment_shader=fs)

        self.scene_p = prog(SCENE_VS, SCENE_FS)
        self.floor_p = prog(FLOOR_VS, FLOOR_FS)
        self.sky_p = prog(QUAD_VS, SKY_FS)
        self.bright_p, self.down_p, self.up_p = prog(QUAD_VS, BRIGHT_FS), prog(QUAD_VS, DOWN_FS), prog(QUAD_VS, UP_FS)
        self.rays_p, self.comp_p = prog(QUAD_VS, RAYS_FS), prog(QUAD_VS, COMPOSITE_FS)
        self.quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4"))
        self.vaos = {name: ctx.vertex_array(p, [(self.quad, "2f", "in_xy")])
                     for name, p in (("sky", self.sky_p), ("floor", self.floor_p), ("bright", self.bright_p),
                                     ("down", self.down_p), ("up", self.up_p), ("rays", self.rays_p),
                                     ("comp", self.comp_p))}

        def tex(w, h, comps=4, dtype="f2"):
            t = ctx.texture((max(w, 1), max(h, 1)), comps, dtype=dtype)
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
            return t

        self.color_t, self.depth_t = tex(self.rw, self.rh), tex(self.rw, self.rh, 1, "f4")
        self.scene_fbo = ctx.framebuffer([self.color_t, self.depth_t], ctx.depth_renderbuffer((self.rw, self.rh)))
        self.color_only_fbo = ctx.framebuffer([self.color_t], self.scene_fbo.depth_attachment)
        half = (self.rw // 2, self.rh // 2)
        self.bright_t = tex(*half)
        self.bright_fbo = ctx.framebuffer([self.bright_t])
        self.levels, w, h = [], *half
        for _ in range(6):
            w, h = max(w // 2, 1), max(h // 2, 1)
            t = tex(w, h)
            self.levels.append((t, ctx.framebuffer([t])))
        self.ups = []
        for t, _ in self.levels[:-1]:
            u = tex(t.width, t.height)
            self.ups.append((u, ctx.framebuffer([u])))
        self.blur_half_t, self.blur_quarter_t = tex(*half), tex(self.rw // 4, self.rh // 4)
        self.blur_half_fbo = ctx.framebuffer([self.blur_half_t])
        self.blur_quarter_fbo = ctx.framebuffer([self.blur_quarter_t])
        self.rays_t = tex(*half)
        self.rays_fbo = ctx.framebuffer([self.rays_t])
        self.out_t = ctx.texture((self.ow, self.oh), 3, dtype="f1")
        self.out_fbo = ctx.framebuffer([self.out_t])
        self.knobs = post_knob_values(knobs or {})
        self._set_look_uniforms()

    def batch(self, mesh, count):
        b = Batch(self.ctx, self.scene_p, mesh, count)
        self.batches.append(b)
        return b

    def _set_look_uniforms(self):
        lk = self.look
        stops = lk["ramp"]
        mid_stop = stops[len(stops) // 2]
        for p in (self.scene_p, self.floor_p, self.sky_p):
            for name, value in (("u_ramp0", stops[0][1]), ("u_ramp1", mid_stop[1]), ("u_ramp2", stops[-1][1]),
                                ("u_ramp_mid", mid_stop[0]), ("u_accent", lk["accent"]),
                                ("u_strength", lk["strength"]), ("u_horizon", lk["horizon"]), ("u_fog", lk["fog"])):
                if name in p:
                    p[name].value = value
        self.sky_p["u_zenith"].value = lk["zenith"]
        key_e, key_c = lk["key"]
        rim_e, rim_c = lk["rim"]
        self.scene_p["u_key_dir"].value = tuple(-euler_dir(50, 0, 35))
        self.scene_p["u_rim_dir"].value = tuple(-euler_dir(-65, 0, 200))
        self.scene_p["u_key_col"].value = tuple(key_e * c for c in key_c)
        self.scene_p["u_rim_col"].value = tuple(rim_e * c for c in rim_c)

    def _full(self, name, fbo):
        fbo.use()
        self.vaos[name].render(moderngl.TRIANGLE_STRIP)

    def draw(self, frame_index):
        ctx, cam, post = self.ctx, self.camera, self.look["post"]
        view = view_matrix(cam["eye"], cam["target"], cam["roll"])
        proj = projection_matrix(cam["lens"], self.rw / self.rh, cam["shift"], far=self.clip_far)
        viewproj = proj @ view
        vp_bytes = viewproj.T.astype("f4").tobytes()
        eye = tuple(float(v) for v in cam["eye"])

        for b in self.batches:
            b.upload()

        # clear, then sky without depth writes (it also fills the DoF depth
        # target with a far value), then the scene with depth testing
        self.scene_fbo.use()
        self.scene_fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.scene_fbo.depth_mask = False
        self.sky_p["u_inv_viewproj"].write(np.linalg.inv(viewproj).T.astype("f4").tobytes())
        self.sky_p["u_eye"].value = eye
        self.vaos["sky"].render(moderngl.TRIANGLE_STRIP)
        self.scene_fbo.depth_mask = True
        ctx.enable(moderngl.DEPTH_TEST)
        self.scene_p["u_viewproj"].write(vp_bytes)
        self.scene_p["u_eye"].value = eye
        for b in self.batches:
            b.render(moderngl.TRIANGLES)

        if self.floor:
            fl = self.floor
            self.color_only_fbo.use()
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            self.color_only_fbo.depth_mask = False
            self.floor_p["u_viewproj"].write(vp_bytes)
            self.floor_p["u_eye"].value = eye
            self.floor_p["u_floor"].value = (fl["center"][0], fl["center"][1], fl["z"], fl["half"])
            self.floor_p["u_floor_step"].value = fl["step"]
            self.floor_p["u_floor_drive"].value = float(np.clip(fl["drive"], 0, 1))
            self.floor_p["u_floor_width"].value = fl.get("width", 1.2)
            self.vaos["floor"].render(moderngl.TRIANGLE_STRIP)
            self.color_only_fbo.depth_mask = True
            ctx.disable(moderngl.BLEND)
        ctx.disable(moderngl.DEPTH_TEST)

        # bright pass + bloom
        self.color_t.use(0)
        self.bright_p["src"].value = 0
        self.bright_p["threshold"].value = post["bloom"][1]
        self._full("bright", self.bright_fbo)
        src = self.bright_t
        for t_lvl, fbo in self.levels:
            src.use(0)
            self.down_p["src"].value = 0
            self.down_p["texel"].value = (1 / src.width, 1 / src.height)
            self._full("down", fbo)
            src = t_lvl
        for i in range(len(self.levels) - 2, -1, -1):
            u_t, fbo = self.ups[i]
            src.use(0)
            self.levels[i][0].use(1)
            self.up_p["src"].value, self.up_p["prev"].value = 0, 1
            self.up_p["texel"].value = (1 / src.width, 1 / src.height)
            self._full("up", fbo)
            src = u_t
        bloom_t = src

        # blurred colour for depth of field
        for src_t, fbo in ((self.color_t, self.blur_half_fbo), (self.blur_half_t, self.blur_quarter_fbo)):
            src_t.use(0)
            self.down_p["src"].value = 0
            self.down_p["texel"].value = (1 / src_t.width, 1 / src_t.height)
            self._full("down", fbo)

        # light shafts from the hero point (skipped when it's behind the camera)
        clip = viewproj @ np.array([*self.hero, 1.0])
        amount = post["rays"] if clip[3] > 0.1 else 0.0
        luv = (clip[0] / clip[3] * 0.5 + 0.5, clip[1] / clip[3] * 0.5 + 0.5) if clip[3] > 0.1 else (0.5, 0.5)
        self.bright_t.use(0)
        self.rays_p["src"].value = 0
        self.rays_p["light_uv"].value = (float(luv[0]), float(luv[1]))
        self.rays_p["amount"].value = amount * self.knobs["rays"]
        self._full("rays", self.rays_fbo)

        # composite at output size
        focus = float(np.linalg.norm(np.asarray(cam["target"]) - np.asarray(cam["eye"])))
        for i, t in enumerate((self.color_t, self.depth_t, self.blur_half_t, self.blur_quarter_t, bloom_t, self.rays_t)):
            t.use(i)
        cp = self.comp_p
        for name, unit in (("color", 0), ("depth", 1), ("blur_half", 2), ("blur_quarter", 3), ("bloom", 4), ("rays", 5)):
            cp[name].value = unit
        cp["focus"].value = focus
        cp["dof"].value = post["dof"] * (cam["lens"] / 35.0) * 1.6 * self.knobs["dof"]
        cp["bloom_amount"].value = post["bloom"][0] * self.knobs["bloom"]
        cp["exposure"].value = self.knobs["exposure"]
        self.scene_p["u_emit_gain"].value = self.knobs["emit"]
        cp["aberration"].value = post["aberration"]
        cp["grain"].value = post["grain"]
        cp["vignette"].value = post["vignette"]
        cp["grade"].value = post["grade"]
        cp["frame"].value = float(frame_index % 997)
        self._full("comp", self.out_fbo)

    def finish(self):
        self.ctx.finish()

    def read(self):
        return self.out_fbo.read(components=3, alignment=1)
