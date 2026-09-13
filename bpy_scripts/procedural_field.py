import bpy, math, sys, time, os, shutil, subprocess

OUT_PATH, WIDTH, HEIGHT, FRAME_COUNT, FPS, GRID_N = sys.argv[1:7]
WIDTH, HEIGHT, FRAME_COUNT, FPS, GRID_N = int(WIDTH), int(HEIGHT), int(FRAME_COUNT), int(FPS), int(GRID_N)
# FRAME_STEP=2 renders "on twos": every other frame is rendered and each is held
# twice at mux time, so the deliverable keeps FRAME_COUNT frames @ FPS.
FRAME_STEP = max(1, int(sys.argv[7])) if len(sys.argv) > 7 else 1

t_start = time.time()
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

SPACING = 0.6
AMPLITUDE = 1.2
WAVELENGTH = 6.0
RIPPLE_PERIOD_FRAMES = 60
ORBIT_PERIOD_FRAMES = 240

base_mesh = bpy.data.meshes.new("sphere_mesh")
tmp = bpy.data.objects.new("tmp", base_mesh)
scene.collection.objects.link(tmp)
bpy.context.view_layer.objects.active = tmp
bpy.ops.object.select_all(action="DESELECT")
tmp.select_set(True)
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.18)
generated = bpy.context.active_object
base_mesh = generated.data
bpy.data.objects.remove(tmp, do_unlink=True)

objs = []
centers = []
half = (GRID_N - 1) / 2.0
for ix in range(GRID_N):
    for iy in range(GRID_N):
        x = (ix - half) * SPACING
        y = (iy - half) * SPACING
        obj = bpy.data.objects.new(f"sph_{ix}_{iy}", base_mesh)
        obj.location = (x, y, 0)
        scene.collection.objects.link(obj)
        objs.append(obj)
        centers.append((x, y, math.sqrt(x * x + y * y)))
bpy.data.objects.remove(generated, do_unlink=True)

mat = bpy.data.materials.new("Field")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
obj_info = nt.nodes.new("ShaderNodeObjectInfo")
sep = nt.nodes.new("ShaderNodeSeparateXYZ")
map_range = nt.nodes.new("ShaderNodeMapRange")
map_range.inputs["From Min"].default_value = -AMPLITUDE
map_range.inputs["From Max"].default_value = AMPLITUDE
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.05, 0.05, 0.6, 1)
ramp.color_ramp.elements[1].color = (1.0, 0.25, 0.05, 1)
emission = nt.nodes.new("ShaderNodeEmission")
emission.inputs["Strength"].default_value = 4.0
output = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(obj_info.outputs["Location"], sep.inputs[0])
nt.links.new(sep.outputs["Z"], map_range.inputs["Value"])
nt.links.new(map_range.outputs["Result"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], emission.inputs["Color"])
nt.links.new(emission.outputs["Emission"], output.inputs["Surface"])
base_mesh.materials.append(mat)

light_data = bpy.data.lights.new("Sun", type="SUN")
light_data.energy = 1.0
light_obj = bpy.data.objects.new("Sun", light_data)
light_obj.rotation_euler = (math.radians(60), 0, math.radians(20))
scene.collection.objects.link(light_obj)

world = bpy.data.worlds.new("W")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.01, 0.012, 0.02, 1)
scene.world = world

orbit_empty = bpy.data.objects.new("Orbit", None)
scene.collection.objects.link(orbit_empty)
cam_data = bpy.data.cameras.new("Cam")
cam_obj = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam_obj)
cam_obj.location = (14, 0, 9)
cam_obj.parent = orbit_empty
scene.camera = cam_obj
target = bpy.data.objects.new("Target", None)
scene.collection.objects.link(target)
con = cam_obj.constraints.new("TRACK_TO")
con.target = target
con.track_axis = "TRACK_NEGATIVE_Z"
con.up_axis = "UP_Y"

def update_frame(scene_):
    f = scene_.frame_current
    t = f / RIPPLE_PERIOD_FRAMES
    for obj, (x, y, dist) in zip(objs, centers):
        z = AMPLITUDE * math.sin(2 * math.pi * (t - dist / WAVELENGTH))
        obj.location.z = z
        s = 0.7 + 0.3 * (z / AMPLITUDE)
        obj.scale = (s, s, s)
    orbit_empty.rotation_euler.z = 2 * math.pi * (f / ORBIT_PERIOD_FRAMES)

bpy.app.handlers.frame_change_pre.append(update_frame)
update_frame(scene)

scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.render.resolution_percentage = 100
scene.render.fps = FPS
scene.frame_start = 1
scene.frame_end = FRAME_COUNT
scene.frame_step = FRAME_STEP
RENDERED_FRAMES = len(range(1, FRAME_COUNT + 1, FRAME_STEP))

frames_dir = OUT_PATH + "_frames"
if os.path.isdir(frames_dir):
    shutil.rmtree(frames_dir)
os.makedirs(frames_dir)
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = frames_dir + "/frame_"

t0 = time.time()
bpy.ops.render.render(animation=True)
t_render = time.time() - t0

t0 = time.time()
# frame_step leaves gaps in the numbering (0001, 0003, ...) -- renumber
# contiguously so ffmpeg's image2 sequence reader doesn't stop at the first gap.
if FRAME_STEP > 1:
    for i, f in enumerate(range(1, FRAME_COUNT + 1, FRAME_STEP)):
        os.rename(f"{frames_dir}/frame_{f:04d}.png", f"{frames_dir}/seq_{i + 1:04d}.png")
    seq_pattern = frames_dir + "/seq_%04d.png"
else:
    seq_pattern = frames_dir + "/frame_%04d.png"
mux_ok = True
try:
    subprocess.check_call([
        "ffmpeg", "-y", "-framerate", f"{FPS}/{FRAME_STEP}",
        "-i", seq_pattern,
        "-r", str(FPS),
        "-pix_fmt", "yuv420p", "-c:v", "libx264",
        OUT_PATH,
    ], stderr=subprocess.DEVNULL)
except Exception:
    mux_ok = False
t_mux = time.time() - t0

t_total = time.time() - t_start
print("===== BPY PROCEDURAL FIELD RESULTS =====")
print(f"instances: {GRID_N * GRID_N}")
print(f"resolution: {WIDTH}x{HEIGHT}")
print(f"frames: {FRAME_COUNT} @ {FPS}fps")
print(f"frame_step: {FRAME_STEP}")
print(f"rendered_frames: {RENDERED_FRAMES}")
print(f"render_time_s: {t_render:.2f}")
print(f"render_time_per_rendered_frame_s: {t_render/RENDERED_FRAMES:.3f}")
print(f"render_time_per_output_frame_s: {t_render/FRAME_COUNT:.3f}")
print(f"mux_time_s: {t_mux:.2f}")
print(f"mux_ok: {mux_ok}")
print(f"total_time_s: {t_total:.2f}")
print(f"output: {OUT_PATH}")
