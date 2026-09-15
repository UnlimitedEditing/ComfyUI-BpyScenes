"""Headless Chromium + three.js render-speed probe.

Usage: python web_probe.py <out.mp4> <render_frames> <capture_frames> <width> <height> <fps>

Installs Playwright's Chromium, then tries several GPU launch configurations
and reports which WebGL renderer each one actually gets (a real GPU vs the
SwiftShader CPU fallback). On the first GPU-backed config it renders the same
test scene as gl_probe.py with three.js post-processing (UnrealBloom, Bokeh
depth of field, Film grain, ACES output), stepping the clock manually, and
measures rendering alone and WebCodecs capture (H.264 encoded inside the page;
only compressed video leaves it). The base64-via-Playwright and binary-POST
capture paths were measured on Graydient at 195 ms and 313 ms per frame and
are no longer run.

Launch configs are tried in two passes: first just to see which WebGL renderer
each gets (NVIDIA preferred over an iGPU), then full timing on the best one.
Prints PROBE_WEB lines as it goes.
"""
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def log(line):
    print(f"PROBE_WEB {line}", flush=True)


THREE = "https://cdn.jsdelivr.net/npm/three@0.169.0"

HTML = """<!doctype html><html><body style="margin:0;background:#000">
<script type="importmap">{"imports": {"three": "%(three)s/build/three.module.js",
                                      "three/addons/": "%(three)s/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { BokehPass } from 'three/addons/postprocessing/BokehPass.js';
import { FilmPass } from 'three/addons/postprocessing/FilmPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
try {
  const W = %(w)d, H = %(h)d, FPS = %(fps)d, N = 20;
  const canvas = document.createElement('canvas'); document.body.appendChild(canvas);
  const renderer = new THREE.WebGLRenderer({canvas, antialias: false, preserveDrawingBuffer: true,
                                            powerPreference: 'high-performance'});
  renderer.setPixelRatio(1); renderer.setSize(W, H, false);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  const gl = renderer.getContext();
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  window.gpu = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, W / H, 0.1, 300); camera.up.set(0, 0, 1);
  const count = N * N;
  const mesh = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 2),
                                       new THREE.MeshBasicMaterial({color: 0xffffff}), count);
  scene.add(mesh);
  const grid = new THREE.GridHelper(160, 80, 0x2a4dff, 0x2a4dff);
  grid.rotation.x = Math.PI / 2; grid.position.z = -2.2; scene.add(grid);
  const pos = [], half = (N - 1) / 2;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const x = (i - half) * 0.55, y = (j - half) * 0.55; pos.push([x, y, Math.hypot(x, y)]);
  }
  const composer = new EffectComposer(renderer);
  composer.setPixelRatio(1); composer.setSize(W, H);
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(W, H), 1.1, 0.5, 0.8));
  composer.addPass(new BokehPass(scene, camera, {focus: 10.0, aperture: 0.004, maxblur: 0.01}));
  composer.addPass(new FilmPass(0.4));
  composer.addPass(new OutputPass());

  const m = new THREE.Object3D(), c = new THREE.Color(), lo = new THREE.Color(0.03, 0.25, 1.0),
        hi = new THREE.Color(1.0, 0.3, 0.04), acc = new THREE.Color(1.0, 0.04, 0.45);
  function stepFrame(f) {
    const t = f / FPS, bp = (t * 128 / 60) %% 1.0, beat = Math.exp(-bp * 60 / 128 / 0.15);
    const amp = 0.4 + 0.6 * (0.5 + 0.5 * Math.sin(t * 0.9)), pr = bp * 60 / 128 * 7.0;
    for (let i = 0; i < count; i++) {
      const [x, y, d] = pos[i], ring = Math.exp(-(((d - pr) / 0.7) ** 2));
      const z = amp * Math.sin(2 * Math.PI * (t * 0.6 - d / 6.0)) + 0.9 * beat * ring;
      const drive = Math.min(Math.max(0.5 + 0.5 * z / (amp + 0.9), 0), 1);
      m.position.set(x, y, z); m.scale.setScalar(0.16 * (0.6 + 0.6 * drive)); m.updateMatrix();
      mesh.setMatrixAt(i, m.matrix);
      c.copy(lo).lerp(hi, drive).lerp(acc, Math.min(ring * beat * 1.5, 1)).multiplyScalar(0.3 + 5 * drive * drive);
      mesh.setColorAt(i, c);
    }
    mesh.instanceMatrix.needsUpdate = true; mesh.instanceColor.needsUpdate = true;
    const a = -0.9 + t * 0.12;
    camera.position.set(10 * Math.cos(a), 10 * Math.sin(a), 2.2 + Math.sin(t * 0.3));
    camera.lookAt(0, 0, -0.4);
    composer.render();
  }
  const px1 = new Uint8Array(4), full = new Uint8Array(W * H * 4);
  window.renderOnly = (start, n) => {
    const t0 = performance.now();
    for (let f = start; f < start + n; f++) { stepFrame(f); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px1); }
    return (performance.now() - t0) / n;
  };
  window.captureBase64 = (f) => {
    stepFrame(f); gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, full);
    let s = ''; const CH = 0x8000;
    for (let i = 0; i < full.length; i += CH) s += String.fromCharCode.apply(null, full.subarray(i, i + CH));
    return btoa(s);
  };
  window.captureFetch = async (start, n) => {
    let rr = 0, post = 0;
    for (let f = start; f < start + n; f++) {
      const t0 = performance.now();
      stepFrame(f); gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, full);
      const t1 = performance.now();
      await fetch('/frame', {method: 'POST', body: full});
      rr += t1 - t0; post += performance.now() - t1;
    }
    return {render_read_ms: rr / n, post_ms: post / n};
  };
  window.captureWebCodecs = async (start, n) => {
    if (!('VideoEncoder' in window)) return {error: 'VideoEncoder unavailable'};
    const chunks = []; let err = null, bytes = 0;
    const enc = new VideoEncoder({
      output: (ch) => { const b = new Uint8Array(ch.byteLength); ch.copyTo(b); chunks.push(b); bytes += b.length; },
      error: (e) => { err = String(e); }});
    const cfg = {codec: 'avc1.640028', width: W, height: H, bitrate: 12000000, framerate: FPS,
                 hardwareAcceleration: 'prefer-hardware', avc: {format: 'annexb'}};
    let sup = await VideoEncoder.isConfigSupported(cfg);
    if (!sup.supported) { cfg.hardwareAcceleration = 'no-preference'; sup = await VideoEncoder.isConfigSupported(cfg); }
    if (!sup.supported) return {error: 'no supported H.264 encoder config'};
    enc.configure(cfg);
    const t0 = performance.now();
    for (let f = start; f < start + n; f++) {
      stepFrame(f);
      const vf = new VideoFrame(canvas, {timestamp: Math.round(f * 1e6 / FPS)});
      enc.encode(vf, {keyFrame: f %% 120 === 0}); vf.close();
      if (enc.encodeQueueSize > 4) await new Promise((r) => setTimeout(r, 0));
    }
    await enc.flush();
    const ms = (performance.now() - t0) / n;
    const all = new Uint8Array(bytes); let o = 0;
    for (const ch of chunks) { all.set(ch, o); o += ch.length; }
    const u0 = performance.now();
    await fetch('/h264', {method: 'POST', body: all});
    return {ms, bytes, upload_ms: performance.now() - u0, err, hw: cfg.hardwareAcceleration};
  };
  window.ready = true;
} catch (e) { window.error = String(e && e.stack || e); }
</script></body></html>"""


class _Sink:
    """State shared with the local HTTP server. Sequential awaits in the page
    keep frames in order."""
    html = b""
    ffmpeg = None
    h264 = None


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(_Sink.html)

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/frame" and _Sink.ffmpeg is not None:
            _Sink.ffmpeg.stdin.write(body)
        elif self.path == "/h264":
            _Sink.h264 = body
        self.send_response(204)
        self.end_headers()


def _first_file(paths):
    return next((p for p in paths if os.path.isfile(p)), None)


# Graydient hosts can expose an AMD iGPU (Mesa radeonsi) that EGL picks by
# default -- a first live run got "AMD Ryzen 9 7950X ... radeonsi" instead of the
# RTX 4090 -- so the NVIDIA EGL vendor / Vulkan ICD are forced via env first.
_NV_EGL = _first_file(["/usr/share/glvnd/egl_vendor.d/10_nvidia.json", "/etc/glvnd/egl_vendor.d/10_nvidia.json"])
_NV_ICD = _first_file(["/usr/share/vulkan/icd.d/nvidia_icd.json", "/etc/vulkan/icd.d/nvidia_icd.json"])
_GL_EGL = ["--use-gl=angle", "--use-angle=gl-egl", "--ignore-gpu-blocklist", "--enable-gpu"]
_VULKAN = ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu"]

CONFIGS = [
    ("chromium, ANGLE GL-EGL, NVIDIA EGL vendor", {"channel": "chromium"}, _GL_EGL,
     {"__EGL_VENDOR_LIBRARY_FILENAMES": _NV_EGL} if _NV_EGL else None),
    ("chromium, ANGLE Vulkan, NVIDIA ICD", {"channel": "chromium"}, _VULKAN,
     {"VK_ICD_FILENAMES": _NV_ICD} if _NV_ICD else None),
    ("chromium, ANGLE GL-EGL", {"channel": "chromium"}, _GL_EGL, {}),
    ("chromium, ANGLE Vulkan", {"channel": "chromium"}, _VULKAN, {}),
    ("headless-shell, ANGLE GL-EGL, NVIDIA EGL vendor", {}, _GL_EGL[:3],
     {"__EGL_VENDOR_LIBRARY_FILENAMES": _NV_EGL} if _NV_EGL else None),
    ("chromium, defaults", {"channel": "chromium"}, [], {}),
]


def main():
    out_path = sys.argv[1]
    render_frames, capture_frames = int(sys.argv[2]), int(sys.argv[3])
    width, height, fps = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])

    t0 = time.time()
    inst = subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], capture_output=True, text=True)
    log(f"playwright install chromium: rc={inst.returncode} in {time.time() - t0:.1f}s"
        + ("" if inst.returncode == 0 else f" err={(inst.stderr or inst.stdout)[-400:]}"))

    from playwright.sync_api import sync_playwright

    _Sink.html = (HTML % {"three": THREE, "w": width, "h": height, "fps": fps}).encode()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/"

    def open_page(p, label, opts, extra, env):
        t0 = time.time()
        launch_env = {**os.environ, **env} if env else None
        try:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", *extra], timeout=45000, env=launch_env,
                                        **opts)
        except Exception as e:  # noqa: BLE001
            log(f"[{label}] launch failed: {str(e).strip().splitlines()[0][:300]}")
            return None
        try:
            page = browser.new_page(viewport={"width": width, "height": height})
            console = []
            page.on("console", lambda msg: console.append(f"{msg.type}: {msg.text}"[:200]))
            page.goto(url)
            page.wait_for_function("window.ready === true || window.error !== undefined", timeout=60000)
            err = page.evaluate("window.error")
            if err:
                log(f"[{label}] page error: {err.splitlines()[0][:200]} console={console[-2:]}")
                browser.close()
                return None
            gpu = page.evaluate("window.gpu")
            log(f"[{label}] launched in {time.time() - t0:.1f}s, WebGL renderer={gpu!r}")
            return browser, page, gpu
        except Exception as e:  # noqa: BLE001
            log(f"[{label}] failed: {str(e).strip().splitlines()[0][:300]}")
            browser.close()
            return None

    def classify(gpu):
        g = gpu.lower()
        if any(s in g for s in ("swiftshader", "llvmpipe", "software", "subzero")):
            return 0
        return 2 if "nvidia" in g else 1

    with sync_playwright() as p:
        # Pass 1: which GPU does each launch config actually get?
        best = None
        for label, opts, extra, env in CONFIGS:
            if env is None:
                log(f"[{label}] skipped: NVIDIA vendor/ICD file not present in container")
                continue
            opened = open_page(p, label, opts, extra, env)
            if not opened:
                continue
            browser, _, gpu = opened
            rank = classify(gpu)
            browser.close()
            if best is None or rank > best[0]:
                best = (rank, label, opts, extra, env, gpu)
            if rank == 2:
                break
        if best is None or best[0] == 0:
            log("no GPU-backed Chromium configuration worked")
            return 2
        rank, label, opts, extra, env, gpu = best
        if rank == 1:
            log(f"WARNING no NVIDIA WebGL renderer found; timing on {gpu!r}")

        # Pass 2: full timing on the best config.
        opened = open_page(p, label, opts, extra, env)
        if not opened:
            return 2
        browser, page, gpu = opened
        try:
            page.evaluate("renderOnly(0, 10)")  # warm-up: shader compile
            ms = page.evaluate(f"renderOnly(10, {render_frames})")
            log(f"[{label}] RESULT render-only (composer.render + 1px sync) {width}x{height} on {gpu!r}: "
                f"{ms:.2f}ms/frame = {1000 / ms:.1f} fps")

            n = max(capture_frames, 1)
            t_cap = time.time()
            r = page.evaluate(f"captureWebCodecs(0, {n})")
            wall = time.time() - t_cap
            if r.get("error") or not _Sink.h264:
                log(f"[{label}] RESULT capture WebCodecs: unavailable ({r.get('error') or r.get('err')})")
                return 3
            raw = out_path + ".h264"
            with open(raw, "wb") as fh:
                fh.write(_Sink.h264)
            mux = subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", raw,
                                  "-c", "copy", out_path], capture_output=True, text=True)
            log(f"[{label}] RESULT capture WebCodecs H.264 ({r.get('hw')}) {width}x{height} {n} frames: "
                f"render+encode={r['ms']:.2f}ms/frame upload={r['upload_ms']:.0f}ms for "
                f"{r['bytes'] / 1e6:.1f}MB wall={wall / n * 1000:.2f}ms/frame ({n / wall:.1f} fps, "
                f"{n / fps / wall:.1f}x realtime) mux_rc={mux.returncode} err={r.get('err')}")
            return 0 if mux.returncode == 0 else 3
        finally:
            browser.close()


if __name__ == "__main__":
    sys.exit(main())
