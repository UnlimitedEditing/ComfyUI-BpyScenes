import bpy, time, sys, os, math, subprocess, shutil
import mathutils

GLTF_PATH, OUT_PATH, WIDTH, HEIGHT, FRAME_COUNT, FPS, ENGINE, SAMPLES = sys.argv[1:9]
WIDTH, HEIGHT, FRAME_COUNT, FPS, SAMPLES = int(WIDTH), int(HEIGHT), int(FRAME_COUNT), int(FPS), int(SAMPLES)

t_start = time.time()
bpy.ops.wm.read_factory_settings(use_empty=True)

t0 = time.time()
bpy.ops.import_scene.gltf(filepath=GLTF_PATH)
t_import = time.time() - t0

scene = bpy.context.scene
min_co = mathutils.Vector((float("inf"),) * 3)
max_co = mathutils.Vector((float("-inf"),) * 3)
for obj in scene.objects:
    if obj.type != "MESH":
        continue
    for corner in obj.bound_box:
        w = obj.matrix_world @ mathutils.Vector(corner)
        min_co.x, min_co.y, min_co.z = min(min_co.x, w.x), min(min_co.y, w.y), min(min_co.z, w.z)
        max_co.x, max_co.y, max_co.z = max(max_co.x, w.x), max(max_co.y, w.y), max(max_co.z, w.z)

center = (min_co + max_co) / 2
size = max_co - min_co
radius = max(size.x, size.y, size.z, 0.01)

cam_data = bpy.data.cameras.new("Cam")
cam_obj = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_distance = radius * 2.2
cam_obj.location = (center.x + cam_distance * 0.7, center.y - cam_distance * 0.9, center.z + cam_distance * 0.5)
cam_obj.rotation_euler = (center - cam_obj.location).to_track_quat("-Z", "Y").to_euler()

light_data = bpy.data.lights.new("Sun", type="SUN")
light_data.energy = 3.0
light_obj = bpy.data.objects.new("Sun", light_data)
light_obj.rotation_euler = (math.radians(50), 0, math.radians(30))
scene.collection.objects.link(light_obj)

scene.render.engine = ENGINE
gpu_found = False
if ENGINE == "CYCLES":
    scene.cycles.samples = SAMPLES
    scene.cycles.device = "GPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for device in prefs.devices:
        if device.type == "CUDA":
            device.use = True
            gpu_found = True
        else:
            device.use = False

scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.render.resolution_percentage = 100
scene.render.fps = FPS
scene.frame_start = 1
scene.frame_end = FRAME_COUNT

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
mux_ok = True
try:
    subprocess.check_call([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", frames_dir + "/frame_%04d.png",
        "-pix_fmt", "yuv420p", "-c:v", "libx264",
        OUT_PATH,
    ], stderr=subprocess.DEVNULL)
except Exception as e:
    mux_ok = False
t_mux = time.time() - t0

t_total = time.time() - t_start
print("===== BPY RENDER RESULTS =====")
print(f"engine: {ENGINE}")
print(f"gpu_device_found: {gpu_found}")
print(f"resolution: {WIDTH}x{HEIGHT}")
print(f"frames: {FRAME_COUNT} @ {FPS}fps")
print(f"import_time_s: {t_import:.2f}")
print(f"render_time_s: {t_render:.2f}")
print(f"render_time_per_frame_s: {t_render/FRAME_COUNT:.3f}")
print(f"mux_time_s: {t_mux:.2f}")
print(f"mux_ok: {mux_ok}")
print(f"total_time_s: {t_total:.2f}")
print(f"output: {OUT_PATH}")
