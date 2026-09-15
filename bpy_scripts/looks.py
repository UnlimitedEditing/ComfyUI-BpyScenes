"""Looks: palette, emission, lighting and world. A look never knows which scene
it's on. It reads only the drive contract from each element's object colour:

  obj.color[0]  drive   0-1 how excited the element is right now
  obj.color[1]  accent  0 = main palette, 1 = the look's accent colour

Depth comes from the shader, not compositor post: elements fade toward the
look's horizon colour with camera distance (aerial perspective), and the world
is a vertical sky gradient.
"""
import math

import bpy

from looks_data import LOOKS  # noqa: E402  (palette/lighting shared with the GL renderer)

def _sock(sockets, identifier):
    return next(s for s in sockets if s.identifier == identifier)


def _ramp(nodes, stops):
    ramp = nodes.new("ShaderNodeValToRGB")
    elems = ramp.color_ramp.elements
    (p0, c0), (p1, c1) = stops[0], stops[-1]
    elems[0].position, elems[0].color = p0, (*c0, 1)
    elems[1].position, elems[1].color = p1, (*c1, 1)
    for pos, col in stops[1:-1]:
        elems.new(pos).color = (*col, 1)
    return ramp


def _map_range(nodes, from_min, from_max, to_min, to_max):
    m = nodes.new("ShaderNodeMapRange")
    m.clamp = True
    m.inputs["From Min"].default_value = from_min
    m.inputs["From Max"].default_value = from_max
    m.inputs["To Min"].default_value = to_min
    m.inputs["To Max"].default_value = to_max
    return m


def _mix_rgb(nodes):
    m = nodes.new("ShaderNodeMix")
    m.data_type = "RGBA"
    return m, _sock(m.inputs, "Factor_Float"), _sock(m.inputs, "A_Color"), _sock(m.inputs, "B_Color"), \
        _sock(m.outputs, "Result_Color")


def build_look(name):
    spec = LOOKS[name]
    scn = bpy.context.scene
    mat = bpy.data.materials.new(f"look_{name}")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, link = nt.nodes, nt.links.new
    nodes.clear()

    info = nodes.new("ShaderNodeObjectInfo")
    sep = nodes.new("ShaderNodeSeparateColor")
    link(info.outputs["Color"], sep.inputs["Color"])
    drive, accent = sep.outputs["Red"], sep.outputs["Green"]

    ramp = _ramp(nodes, spec["ramp"])
    link(drive, ramp.inputs["Fac"])

    # accent elements: accent colour scaled up with drive
    acc_scale = _map_range(nodes, 0.0, 1.0, 0.25, 1.0)
    link(drive, acc_scale.inputs["Value"])
    acc_col, f_in, a_in, b_in, acc_out = _mix_rgb(nodes)
    acc_col.blend_type = "MULTIPLY"
    f_in.default_value = 1.0
    a_in.default_value = (*spec["accent"], 1)
    link(acc_scale.outputs["Result"], b_in)
    pick, f_in, a_in, b_in, colour = _mix_rgb(nodes)
    link(accent, f_in)
    link(ramp.outputs["Color"], a_in)
    link(acc_out, b_in)

    # aerial perspective toward the horizon colour
    cam = nodes.new("ShaderNodeCameraData")
    near, far, fog_max = spec["fog"]
    fog = _map_range(nodes, near, far, 0.0, fog_max)
    link(cam.outputs["View Distance"], fog.inputs["Value"])
    fogged, f_in, a_in, b_in, fogged_out = _mix_rgb(nodes)
    link(fog.outputs["Result"], f_in)
    link(colour, a_in)
    b_in.default_value = (*spec["horizon"], 1)

    strength = _map_range(nodes, 0.0, 1.0, *spec["strength"])
    link(drive, strength.inputs["Value"])
    fade = nodes.new("ShaderNodeMath")
    fade.operation = "SUBTRACT"
    fade.inputs[0].default_value = 1.0
    link(fog.outputs["Result"], fade.inputs[1])
    strength_fogged = nodes.new("ShaderNodeMath")
    strength_fogged.operation = "MULTIPLY"
    link(strength.outputs["Result"], strength_fogged.inputs[0])
    link(fade.outputs["Value"], strength_fogged.inputs[1])

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.4
    link(fogged_out, bsdf.inputs["Base Color"])
    link(fogged_out, bsdf.inputs["Emission Color"])
    link(strength_fogged.outputs["Value"], bsdf.inputs["Emission Strength"])
    out = nodes.new("ShaderNodeOutputMaterial")
    link(bsdf.outputs["BSDF"], out.inputs["Surface"])

    world = bpy.data.worlds.new(f"world_{name}")
    world.use_nodes = True
    wn = world.node_tree.nodes
    wl = world.node_tree.links.new
    coord = wn.new("ShaderNodeTexCoord")
    xyz = wn.new("ShaderNodeSeparateXYZ")
    wl(coord.outputs["Generated"], xyz.inputs["Vector"])
    # Cameras mostly look near the horizon, so the glow must be a narrow band or
    # it fills the whole frame; below the horizon falls back to near-zenith dark
    # so the floor grid reads against it.
    height = _map_range(wn, -0.3, 0.3, 0.0, 1.0)
    wl(xyz.outputs["Z"], height.inputs["Value"])
    ground = tuple(c * 0.5 for c in spec["zenith"])
    sky = _ramp(wn, [(0.0, ground), (0.47, spec["horizon"]), (0.53, spec["horizon"]), (0.75, spec["zenith"])])
    wl(height.outputs["Result"], sky.inputs["Fac"])
    wl(sky.outputs["Color"], wn["Background"].inputs["Color"])
    scn.world = world

    for label, (energy, colour), rot in (("key", spec["key"], (50, 0, 35)),
                                          ("rim", spec["rim"], (-65, 0, 200))):
        light = bpy.data.objects.new(f"look_{label}", bpy.data.lights.new(f"look_{label}", type="SUN"))
        light.data.energy = energy
        light.data.color = colour
        light.rotation_euler = tuple(math.radians(a) for a in rot)
        scn.collection.objects.link(light)

    scn.view_settings.view_transform = "Standard"
    return mat
