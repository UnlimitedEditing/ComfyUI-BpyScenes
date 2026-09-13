"""Isolated Blender runtime for ComfyUI nodes.

bpy's pip wheels only exist for CPython 3.11 and 3.13+, never 3.12 -- which is
what Graydient's ComfyUI runs. So bpy is never imported in-process: a portable
Python 3.13 (astral-sh/python-build-standalone) is provisioned inside the
container, bpy is pip-installed into it, and scene scripts in bpy_scripts/ run
as subprocesses.
"""
import asyncio
import os
import signal
import tarfile
import tempfile
import time
import urllib.request

PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    "20260901/cpython-3.13.15%2B20260901-x86_64-unknown-linux-gnu-install_only.tar.gz"
)

WORK_DIR = os.path.join(tempfile.gettempdir(), "bpy_scenes")
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpy_scripts")


def script_path(name):
    return os.path.join(SCRIPTS_DIR, name)


async def run_subprocess(args, stream_prefix=None):
    """Runs a subprocess without blocking the asyncio event loop.

    A synchronous subprocess.run() inside a node FUNCTION stalls ComfyUI's
    whole async executor for the child's lifetime; on Graydient that made
    finished jobs report timed_out with no error. Each child gets its own
    process group, killed afterwards so no orphaned Blender/ffmpeg children
    keep the container looking busy.

    With stream_prefix, stdout is echoed line by line with a wall-clock stamp
    as it arrives, so a job killed by the platform timeout still leaves
    progress and timing in its log (Blender's per-frame "Saved:" lines are
    dropped from the echo and the returned text)."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
        limit=1 << 20,
    )
    out_lines = []
    try:
        stderr_task = asyncio.create_task(proc.stderr.read())
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode(errors="replace").rstrip("\n")
            if "| Saved:" in line:
                continue
            out_lines.append(line)
            if stream_prefix:
                print(f"[{stream_prefix} {time.strftime('%H:%M:%S')}] {line}", flush=True)
        stderr = await stderr_task
        await proc.wait()
        stdout = "\n".join(out_lines).encode()
    finally:
        sig = signal.SIGKILL if proc.returncode is None else signal.SIGTERM
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
    return proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def ensure_bpy_python(log):
    """Returns the path to a Python 3.13 binary that can import bpy,
    provisioning it on first use. Reused across jobs on a warm container."""
    os.makedirs(WORK_DIR, exist_ok=True)
    py_dir = os.path.join(WORK_DIR, "python3.13")
    py_bin = os.path.join(py_dir, "bin", "python3.13")

    t0 = time.time()
    cached = os.path.isfile(py_bin)
    if not cached:
        tar_path = os.path.join(WORK_DIR, "python.tar.gz")
        await asyncio.to_thread(urllib.request.urlretrieve, PYTHON_URL, tar_path)
        with tarfile.open(tar_path) as tf:
            tf.extractall(WORK_DIR)
        extracted = os.path.join(WORK_DIR, "python")
        if os.path.isdir(extracted) and not os.path.isdir(py_dir):
            os.rename(extracted, py_dir)
    log(f"python_provision_time_s: {time.time() - t0:.2f} (cached: {cached})")

    t0 = time.time()
    rc, _, _ = await run_subprocess([py_bin, "-c", "import bpy"])
    already = rc == 0
    if not already:
        rc, _, err = await run_subprocess([py_bin, "-m", "pip", "install", "--quiet", "bpy"])
        if rc != 0:
            raise RuntimeError(f"bpy install failed (rc={rc}):\n{err}")
    log(f"bpy_install_time_s: {time.time() - t0:.2f} (already_installed: {already})")
    return py_bin
