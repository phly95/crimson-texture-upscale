"""klein_server.py - FLUX.2 klein 9B kept loaded in stable-diffusion.cpp's sd-server (127.0.0.1:18234), so a redraw does not reload the models.
Same models, steps (4), cfg (1.0) and sampler (euler) as klein.sh: output was verified bit-identical to sd-cli for the same seed (terpat01, seeds 1 and 2).
Warm request ~15.6 s vs ~20-22 s for a fresh sd-cli process.  It holds ~9.5 GB VRAM: stop it before running anything else on the GPU (Strata, klein.sh).
CLI: python klein_server.py start|stop|status"""
import base64, json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

PORT = 18234
URL = f"http://127.0.0.1:{PORT}"
K = Path.home() / "workspace/flux2-klein/models"
B = Path.home() / "workspace/qwen-image/bin"
COUNT = {"n": 0}


def _call(path, body=None, timeout=30):
    req = urllib.request.Request(URL + path, json.dumps(body).encode() if body is not None else None, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def up():
    try:
        return _call("/sdcpp/v1/capabilities", timeout=3).get("model", {}).get("stem") == "flux-2-klein-9b-Q8_0"
    except Exception:                                          # noqa: BLE001
        return False


def pids():
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()
    return [int(l.split()[0]) for l in out if "/sd-server" in l and f"--listen-port {PORT}" in l]


def start(log_path, wait=180):
    if up():
        return
    stop()
    env = dict(os.environ, LD_LIBRARY_PATH=str(B))
    cmd = [str(B / "sd-server"), "--diffusion-model", str(K / "flux-2-klein-9b-Q8_0.gguf"), "--vae", str(K / "flux2-vae.safetensors"),
           "--llm", str(K / "Qwen3-8B-Q8_0.gguf"), "--cfg-scale", "1.0", "--sampling-method", "euler", "--steps", "4", "--listen-port", str(PORT)]
    with open(log_path, "ab") as lf:
        subprocess.Popen(cmd, stdout=lf, stderr=lf, env=env, start_new_session=True)
    t = time.time()
    while time.time() - t < wait:
        if up():
            COUNT["n"] = 0
            return
        time.sleep(1)
    raise RuntimeError("klein server did not come up")


def stop():
    for p in pids():
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    t = time.time()
    while pids() and time.time() - t < 30:
        time.sleep(0.5)
    for p in pids():
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass
    time.sleep(2)
    COUNT["n"] = 0


def submit(ref, prompt, W, H, seed):
    """queue one redraw on the server and return the job id (the server runs its queue one job at a time, so queued jobs keep the GPU busy)"""
    b64 = base64.b64encode(Path(ref).read_bytes()).decode()
    r = _call("/sdcpp/v1/img_gen", {"prompt": prompt, "width": int(W), "height": int(H), "seed": int(seed), "ref_images": [b64], "output_format": "png"}, timeout=30)
    return r["id"]


def collect(jid, out, timeout=150, tick=None):
    """wait for job JID and write its image to OUT; returns 0, raises on server trouble"""
    t = time.time()
    while True:
        j = _call(f"/sdcpp/v1/jobs/{jid}", timeout=30)
        if j.get("status") == "completed":
            break
        if j.get("status") in ("failed", "cancelled", "error") or j.get("error"):
            raise RuntimeError(f"klein server job {j.get('status')}: {j.get('error')}")
        if time.time() - t > timeout:
            try:
                _call(f"/sdcpp/v1/jobs/{jid}/cancel", {}, timeout=10)
            except Exception:                                  # noqa: BLE001
                pass
            raise TimeoutError(f"klein server job took more than {timeout} s")
        if tick:
            tick()
        time.sleep(0.5)
    imgs = (j.get("result") or {}).get("images") or []
    if not imgs:
        raise RuntimeError("klein server returned no image")
    Path(out).write_bytes(base64.b64decode(imgs[0].get("b64_json") or imgs[0]))
    COUNT["n"] += 1
    return 0


def gen(ref, out, prompt, W, H, seed, timeout=150, tick=None):
    return collect(submit(ref, prompt, W, H, seed), out, timeout, tick)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "start":
        start(Path.home() / "workspace/csvm-hd/pipeline/klein_server.log"); print("up")
    elif cmd == "stop":
        stop(); print("stopped")
    else:
        print("up" if up() else "down", pids())
