"""colormatch.py UNIT_DIR [seed ...]   writes klein_s<seed>_cm.png next to the raw klein result.

Frequency separation in Lab: the redraw keeps all of its fine detail, but its broad colour and brightness (everything
coarser than about 0.6 original pixels) are pulled back to the original's.  Cut-outs are handled with normalised
convolution inside the shape only (the green screen never leaks into the blur).  The correction is clipped so a redraw
whose layout drifted cannot be dragged into ghosts.

    from colormatch import colormatch
    colormatch(Path("pipeline/units/<uid>"), 1)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage import color as skc

CLIP = np.array([20.0, 16.0, 16.0], np.float32)       # max correction in L, a, b


def colormatch(d, seed, strength=0.85, sigma_orig_px=0.6, sigma_frac=0.03):
    d = Path(d)
    kl = Image.open(d / f"klein_s{seed}.png").convert("RGB")
    W, H = kl.size
    orig = Image.open(d / "orig.png").convert("RGBA")
    ow = orig.size[0]
    k = np.asarray(kl).astype(np.float32) / 255
    a_o = np.asarray(orig.getchannel("A").resize((W, H), Image.BICUBIC)).astype(np.float32) / 255
    cut = (np.asarray(orig.getchannel("A")) < 128).mean() > 0.02 or (np.asarray(orig.getchannel("A")) < 250).mean() > 0.02
    g = k[..., 1] - np.maximum(k[..., 0], k[..., 2])
    bg = (g > 0.27) & (k[..., 1] > 0.58)                           # green screen
    mo = (a_o >= 0.5) if cut else np.ones((H, W), bool)
    m = ((~bg) & mo).astype(np.float32)
    m = cv2.erode(m, np.ones((3, 3), np.uint8))                    # stay off the cut-out fringe
    o = np.asarray(orig.convert("RGB").resize((W, H), Image.BICUBIC)).astype(np.float32) / 255
    lab_k, lab_o = skc.rgb2lab(k).astype(np.float32), skc.rgb2lab(o).astype(np.float32)
    sigma = max(1.5, sigma_orig_px * W / ow, sigma_frac * max(W, H))   # never finer than 3% of the frame: only broad tone is matched

    def nblur(x):
        num = cv2.GaussianBlur(x * m[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(m, (0, 0), sigma)[..., None]
        return num / np.maximum(den, 1e-3)

    delta = np.clip(nblur(lab_o) - nblur(lab_k), -CLIP, CLIP)
    out_lab = lab_k + strength * delta * m[..., None]
    out = (skc.lab2rgb(out_lab.astype(np.float64)) * 255).clip(0, 255).astype(np.uint8)
    w = cv2.GaussianBlur(m, (0, 0), 1.0)[..., None]                 # soft edge of the corrected area
    out = (out * w + np.asarray(kl) * (1 - w)).astype(np.uint8)
    Image.fromarray(out).save(d / f"klein_s{seed}_cm.png")
    return d / f"klein_s{seed}_cm.png"


if __name__ == "__main__":
    d = Path(sys.argv[1])
    seeds = [int(x) for x in sys.argv[2:]] or [int(p.stem.split("_s")[1]) for p in sorted(d.glob("klein_s?.png"))]
    for s in seeds:
        print(colormatch(d, s))
