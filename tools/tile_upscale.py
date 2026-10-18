"""tile_upscale.py SRC OUT [--scale 4] [--hint TEXT] [--seed N] [--tile 1024] [--overlap 256] [--keep DIR]: tiled FLUX.2 klein upscale of a large picture (fullscreen UI backgrounds, maps).

Each stage doubles the size: Lanczos 2x -> overlapping 1024 px tiles -> klein *edit* (the tile is the reference image, so layout is kept while new fine detail is drawn)
-> per-tile colour lock (the tile's broad Lab tone is pulled back to the Lanczos tile, so brightness/hue cannot drift) -> cosine-feathered blend of the overlaps.
Two stages = 4x. Flat tiles (nearly one colour) are not sent to the model. Needs the klein server (tools/qwen/klein_server.py start).
Prints per-stage timing and drift metrics (broad-tone dE of the result against the plain Lanczos upscale).  Never touches pipeline/ units."""
import argparse, math, sys, time
from pathlib import Path

import warnings
import cv2
import numpy as np
from PIL import Image
from skimage import color as skc

warnings.filterwarnings('ignore', category=UserWarning)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "qwen"))
import klein_server as ks                                                         # noqa: E402

SIGMA_L, SIGMA_AB = 8.0, 24.0
HF_CLIP, HF_MIN, HF_K = False, 3.0, 2.5
BASE_PROMPT = ("Restore this image as a crisp, sharp, high-resolution version of itself. Keep every shape, colour, position and all lettering exactly the same; "
               "add only fine, realistic surface detail and remove blur and compression noise.")


def locked_gen(ref, out, prompt, T, seed, tries=5):
    return locked_gen_wh(ref, out, prompt, T, T, seed, tries)


def locked_gen_wh(ref, out, prompt, W, H, seed, tries=5):
    """one klein redraw, serialised across processes (a second client at the same time makes the server answer 'no results') and retried"""
    import fcntl
    with open("/tmp/klein_tile.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for k in range(tries):
            try:
                if not ks.up():
                    ks.start(ROOT / "pipeline" / "klein_server.log")
                return ks.gen(ref, out, prompt, W, H, seed, timeout=600)
            except (RuntimeError, TimeoutError, OSError) as e:
                if k == tries - 1:
                    raise
                print("klein retry:", e, flush=True); time.sleep(20)


def positions(L, T, overlap):
    if L <= T:
        return [0]
    n = math.ceil((L - T) / (T - overlap)) + 1
    return [round(i * (L - T) / (n - 1)) for i in range(n)]


def colour_lock(gen, ref, sigma_l, sigma_ab, k_l=0.9):
    """pull the broad Lab tone of gen back to ref (float RGB 0..1): lightness at a fine scale (so the redraw cannot invent its own large shading/creases, only detail finer
    than sigma_l), colour at a coarser one; fine texture of gen is untouched"""
    lg, lr = skc.rgb2lab(gen).astype(np.float32), skc.rgb2lab(ref).astype(np.float32)
    dl = cv2.GaussianBlur(lr[..., :1], (0, 0), sigma_l)[..., None] - cv2.GaussianBlur(lg[..., :1], (0, 0), sigma_l)[..., None]
    dab = cv2.GaussianBlur(lr[..., 1:], (0, 0), sigma_ab) - cv2.GaussianBlur(lg[..., 1:], (0, 0), sigma_ab)
    d = np.concatenate([k_l * dl, dab], -1)
    out = lg + d
    if HF_CLIP:                                  # soft-limit the redraw's fine lightness detail to what the source could plausibly hold: grain stays, invented strokes/labels flatten
        L, rl = out[..., 0], lr[..., 0]
        low = cv2.GaussianBlur(L, (0, 0), 3.0)
        rhf = rl - cv2.GaussianBlur(rl, (0, 0), 3.0)
        A = np.maximum(HF_MIN, HF_K * np.sqrt(cv2.GaussianBlur(rhf * rhf, (0, 0), 10.0)))
        out[..., 0] = low + A * np.tanh((L - low) / A)
    return (skc.lab2rgb(out.astype(np.float64)).clip(0, 1)).astype(np.float32)


def weight(h, w, ox0, ox1, oy0, oy1):
    """feather ramps only on the sides that overlap a neighbour"""
    wx, wy = np.ones(w, np.float32), np.ones(h, np.float32)
    if ox0: wx[:ox0] = np.sin(np.linspace(0, np.pi / 2, ox0, endpoint=False)) ** 2 + 1e-3
    if ox1: wx[-ox1:] = np.cos(np.linspace(0, np.pi / 2, ox1, endpoint=False)) ** 2 + 1e-3
    if oy0: wy[:oy0] = np.sin(np.linspace(0, np.pi / 2, oy0, endpoint=False)) ** 2 + 1e-3
    if oy1: wy[-oy1:] = np.cos(np.linspace(0, np.pi / 2, oy1, endpoint=False)) ** 2 + 1e-3
    return wy[:, None] * wx[None, :]


def stage(img, prompt, seed, T, overlap, keep=None, tag=""):
    W, H = img.size[0] * 2, img.size[1] * 2
    base = img.resize((W, H), Image.LANCZOS)
    b = np.asarray(base).astype(np.float32) / 255
    T = min(T, W, H)
    xs, ys = positions(W, T, overlap), positions(H, T, overlap)
    acc, wacc = np.zeros_like(b), np.zeros((H, W), np.float32)
    tmp = Path(keep or "/tmp") / f"_tile_{tag}.png"
    n = sent = 0; t0 = time.time()
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            ref = b[y:y + T, x:x + T]
            n += 1
            if ref.std(axis=(0, 1)).max() < 0.012:                     # flat: nothing to add
                out = ref
            else:
                Image.fromarray((ref * 255 + .5).astype(np.uint8)).save(tmp)
                out_p = tmp.with_name(tmp.stem + "_out.png")
                locked_gen(tmp, out_p, prompt, T, seed + n)
                g = np.asarray(Image.open(out_p).convert("RGB").resize((T, T), Image.LANCZOS)).astype(np.float32) / 255
                out = colour_lock(g, ref, SIGMA_L, SIGMA_AB); sent += 1
                if keep:
                    Image.fromarray((out * 255 + .5).astype(np.uint8)).save(Path(keep) / f"{tag}_t{yi}{xi}.png")
            ox0 = (xs[xi - 1] + T - x) if xi else 0; ox1 = (x + T - xs[xi + 1]) if xi + 1 < len(xs) else 0
            oy0 = (ys[yi - 1] + T - y) if yi else 0; oy1 = (y + T - ys[yi + 1]) if yi + 1 < len(ys) else 0
            w = weight(T, T, ox0, ox1, oy0, oy1)
            acc[y:y + T, x:x + T] += out * w[..., None]; wacc[y:y + T, x:x + T] += w
    res = acc / np.maximum(wacc, 1e-6)[..., None]
    for p in (tmp, tmp.with_name(tmp.stem + "_out.png")):
        p.unlink(missing_ok=True)
    print(f"  stage {tag}: {W}x{H}, {len(xs)}x{len(ys)} tiles ({sent} sent to klein) in {time.time() - t0:.0f}s", flush=True)
    return Image.fromarray((res.clip(0, 1) * 255 + .5).astype(np.uint8)), base


def drift(res, base):
    a = skc.rgb2lab(np.asarray(res.resize((res.size[0] // 8, res.size[1] // 8), Image.BOX)) / 255.0)
    b = skc.rgb2lab(np.asarray(base.resize((base.size[0] // 8, base.size[1] // 8), Image.BOX)) / 255.0)
    return float(np.sqrt(((a - b) ** 2).sum(-1)).mean())


def upscale(src, out, scale=4, hint="", seed=100, T=1024, overlap=256, keep=None, sigma_l=None, sigma_ab=None, hf_clip=False):
    global SIGMA_L, SIGMA_AB, HF_CLIP
    HF_CLIP = hf_clip
    SIGMA_L, SIGMA_AB = sigma_l or SIGMA_L, sigma_ab or SIGMA_AB
    img = Image.open(src).convert("RGB")
    prompt = BASE_PROMPT + (" " + hint if hint else "")
    if keep:
        Path(keep).mkdir(parents=True, exist_ok=True)
    cur, k, t0 = img, 1, time.time()
    while k < scale:
        cur, base = stage(cur, prompt, seed, T, overlap, keep, tag=f"s{k * 2}")
        k *= 2
    final_base = img.resize(cur.size, Image.LANCZOS)
    print(f"done {Path(src).name} -> {cur.size} in {time.time() - t0:.0f}s; broad-tone drift vs Lanczos dE {drift(cur, final_base):.2f}", flush=True)
    cur.save(out)
    return cur


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--scale", type=int, default=4); ap.add_argument("--hint", default="")
    ap.add_argument("--seed", type=int, default=100); ap.add_argument("--tile", type=int, default=1024); ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--keep"); ap.add_argument("--hf-clip", action="store_true"); ap.add_argument("--sigma-l", type=float); ap.add_argument("--sigma-ab", type=float)
    a = ap.parse_args()
    if not ks.up():
        ks.start(ROOT / "pipeline" / "klein_server.log")
    upscale(a.src, a.out, a.scale, a.hint, a.seed, a.tile, a.overlap, a.keep, a.sigma_l, a.sigma_ab, a.hf_clip)
