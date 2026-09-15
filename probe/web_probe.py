"""Headless Chromium + three.js render-speed probe.

Usage: python web_probe.py <out.mp4> <render_frames> <capture_frames> <width> <height> <fps>

Installs Playwright's Chromium, then tries several GPU launch configurations
and reports which WebGL renderer each one actually gets (a real GPU vs the
SwiftShader CPU fallback). On the first GPU-backed config it renders the same
test scene as gl_probe.py with three.js post-processing (UnrealBloom, Bokeh
depth of field, Film grain, ACES output), stepping the clock manually, and
measures rendering alone plus three ways of getting frames out:
  base64     readPixels -> base64 string -> Playwright -> Python (naive)
  fetch      readPixels -> binary POST to a local HTTP server -> ffmpeg
  webcodecs  H.264 encoded inside the page; only compressed video leaves it
Prints PROBE_WEB lines as it goes.
"""
import base64
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


CONFIGS = [
    ("chromium, ANGLE Vulkan", {"channel": "chromium"},
     ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu"]),
    ("chromium, ANGLE GL-EGL", {"channel": "chromium"},
     ["--use-gl=angle", "--use-angle=gl-egl", "--ignore-gpu-blocklist", "--enable-gpu"]),
    ("headless-shell, ANGLE Vulkan", {}, ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist"]),
    ("headless-shell, ANGLE GL-EGL", {}, ["--use-gl=angle", "--use-angle=gl-egl", "--ignore-gpu-blocklist"]),
    ("chromium, defaults", {"channel": "chromium"}, []),
]


def _encoder_args():
    probe = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=320x240:rate=24:duration=0.2",
                            "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True)
    if probe.returncode == 0:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]


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
    enc = _encoder_args()

    def raw_sink(path):
        return subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                                 "-s", f"{width}x{height}", "-framerate", str(fps), "-i", "-", "-vf", "vflip",
                                 *enc, "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)

    with sync_playwright() as p:
        for label, opts, extra in CONFIGS:
            t0 = time.time()
            try:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox", *extra], timeout=45000, **opts)
            except Exception as e:  # noqa: BLE001
                log(f"[{label}] launch failed: {str(e).strip().splitlines()[0][:300]}")
                continue
            try:
                page = browser.new_page(viewport={"width": width, "height": height})
                console = []
                page.on("console", lambda msg: console.append(f"{msg.type}: {msg.text}"[:200]))
                page.goto(url)
                page.wait_for_function("window.ready === true || window.error !== undefined", timeout=60000)
                err = page.evaluate("window.error")
                if err:
                    log(f"[{label}] page error: {err[:400]} console={console[-3:]}")
                    continue
                gpu = page.evaluate("window.gpu")
                software = any(s in gpu.lower() for s in ("swiftshader", "llvmpipe", "software", "subzero"))
                log(f"[{label}] launched in {time.time() - t0:.1f}s, WebGL renderer={gpu!r} software={software}")
                if software:
                    continue

                page.evaluate("renderOnly(0, 10)")  # warm-up: shader compile
                ms = page.evaluate(f"renderOnly(10, {render_frames})")
                log(f"[{label}] RESULT render-only (composer.render + 1px sync) {width}x{height}: "
                    f"{ms:.2f}ms/frame = {1000 / ms:.1f} fps")

                n = max(capture_frames, 1)
                few = min(n, 20)
                ff = raw_sink(out_path + ".base64.mp4")
                t_cap = time.time()
                for f in range(few):
                    ff.stdin.write(base64.b64decode(page.evaluate(f"captureBase64({f})")))
                ff.stdin.close()
                ff.wait()
                log(f"[{label}] RESULT capture base64 via Playwright: {(time.time() - t_cap) / few * 1000:.1f}ms/frame")

                # Capped: locally this path measured ~450 ms/frame, so a full run could eat minutes.
                n_fetch = min(n, 60)
                _Sink.ffmpeg = raw_sink(out_path + ".fetch.mp4")
                t_cap = time.time()
                r = page.evaluate(f"captureFetch(0, {n_fetch})")
                _Sink.ffmpeg.stdin.close()
                rc_fetch = _Sink.ffmpeg.wait()
                _Sink.ffmpeg = None
                wall = time.time() - t_cap
                log(f"[{label}] RESULT capture fetch-POST raw RGBA {width}x{height} {n_fetch} frames: "
                    f"render+readPixels={r['render_read_ms']:.2f}ms post+ffmpeg={r['post_ms']:.2f}ms "
                    f"wall={wall / n_fetch * 1000:.2f}ms/frame ({n_fetch / wall:.1f} fps) ffmpeg_rc={rc_fetch}")

                t_cap = time.time()
                r = page.evaluate(f"captureWebCodecs(0, {n})")
                wall = time.time() - t_cap
                chosen, rc = out_path + ".fetch.mp4", rc_fetch
                if r.get("error") or not _Sink.h264:
                    log(f"[{label}] RESULT capture WebCodecs: unavailable ({r.get('error') or r.get('err')})")
                else:
                    raw = out_path + ".h264"
                    with open(raw, "wb") as fh:
                        fh.write(_Sink.h264)
                    mux = subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", raw,
                                          "-c", "copy", out_path + ".webcodecs.mp4"], capture_output=True, text=True)
                    log(f"[{label}] RESULT capture WebCodecs H.264 ({r.get('hw')}) {width}x{height} {n} frames: "
                        f"render+encode={r['ms']:.2f}ms/frame upload={r['upload_ms']:.0f}ms for "
                        f"{r['bytes'] / 1e6:.1f}MB wall={wall / n * 1000:.2f}ms/frame ({n / wall:.1f} fps) "
                        f"mux_rc={mux.returncode} err={r.get('err')}")
                    if mux.returncode == 0:
                        chosen, rc = out_path + ".webcodecs.mp4", 0
                if os.path.isfile(chosen):
                    os.replace(chosen, out_path)
                return 0 if rc == 0 else 3
            except Exception as e:  # noqa: BLE001
                log(f"[{label}] failed: {str(e).strip().splitlines()[0][:300]}")
            finally:
                browser.close()
    log("no GPU-backed Chromium configuration worked")
    return 2


if __name__ == "__main__":
    sys.exit(main())
