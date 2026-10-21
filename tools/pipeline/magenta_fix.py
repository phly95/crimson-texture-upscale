"""magenta_fix.py UID [UID ...]   re-keys and colour-matches the magenta-screen redraws of units (units/<uid>/magenta/klein_s<N>.png -> klein_s<N>_rgba.png).

render_magenta keyed with neutralize=True for non-green subjects, which whitens highlights, and it skipped the Lab colour match the green redraws get
(colormatch.py), so the redraws came out lighter and whiter than the originals (palm fronds, fir needles). This re-keys without neutralize and then pulls the
broad colour and brightness of the foliage back to the original's (same frequency separation as colormatch.py, mask = the keyed shape). The previous rgba is kept as
klein_s<N>_rgba.prefix.png once."""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage import color as skc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from magentakey import magenta_key

ROOT = Path(__file__).resolve().parents[2]
CLIP = np.array([20.0, 16.0, 16.0], np.float32)


def green_subject(orig):
    o = np.asarray(orig).astype(np.float32) / 255
    vis = o[..., 3] > 0.3
    return bool(vis.any() and ((o[..., 1] > np.maximum(o[..., 0], o[..., 2]) + 0.012)[vis]).mean() > 0.3)


def fix_seed(d, seed, strength=0.85, sigma_orig_px=0.6, sigma_frac=0.03):
    d = Path(d); r = d / "magenta"
    raw = np.asarray(Image.open(r / f"klein_s{seed}.png").convert("RGB")).astype(np.float32)
    orig = Image.open(d / "orig.png").convert("RGBA")
    arr, _, _ = magenta_key(raw, neutralize=False, green_subject=green_subject(orig))
    H, W = arr.shape[:2]
    a_k = arr[..., 3].astype(np.float32) / 255
    a_o = np.asarray(orig.getchannel("A").resize((W, H), Image.BICUBIC)).astype(np.float32) / 255
    m = cv2.erode(((a_k >= 0.5) & (a_o >= 0.3)).astype(np.float32), np.ones((3, 3), np.uint8))
    mo = cv2.erode((a_o >= 0.5).astype(np.float32), np.ones((3, 3), np.uint8))
    k = arr[..., :3].astype(np.float32) / 255
    o = np.asarray(orig.convert("RGB").resize((W, H), Image.BICUBIC)).astype(np.float32) / 255
    lab_k, lab_o = skc.rgb2lab(k).astype(np.float32), skc.rgb2lab(o).astype(np.float32)
    sigma = max(1.5, sigma_orig_px * W / orig.size[0], sigma_frac * max(W, H))

    def nblur(x, mask):
        num = cv2.GaussianBlur(x * mask[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(mask, (0, 0), sigma)[..., None]
        return num / np.maximum(den, 1e-3)

    delta = np.clip(nblur(lab_o, mo) - nblur(lab_k, m), -CLIP, CLIP)
    out = (skc.lab2rgb((lab_k + strength * delta * m[..., None]).astype(np.float64)) * 255).clip(0, 255).astype(np.uint8)
    w = cv2.GaussianBlur(m, (0, 0), 1.0)[..., None]
    rgb = (out * w + arr[..., :3] * (1 - w)).astype(np.uint8)
    dst = r / f"klein_s{seed}_rgba.png"
    bak = r / f"klein_s{seed}_rgba.prefix.png"
    if dst.exists() and not bak.exists():
        shutil.copy2(dst, bak)
    Image.fromarray(np.dstack([rgb, arr[..., 3]]), "RGBA").save(dst)
    return dst


if __name__ == "__main__":
    for uid in sys.argv[1:]:
        for s in (1, 2, 3, 4):
            if (ROOT / "pipeline" / "units" / uid / "magenta" / f"klein_s{s}.png").exists():
                print(fix_seed(ROOT / "pipeline" / "units" / uid, s))
