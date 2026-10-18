"""suppress_invented.py SRC REDRAW4X OUT: remove structure the redraw invented where the SOURCE is flat (stray lines/grids/marks in near-black or plain areas).
Flat area = the source's local lightness variation (after upscaling) is tiny, eroded so real lines at its border survive. There the lightness becomes the source's own smooth tone
plus only the finest grain of the redraw (soft-limited), colour from the source. Everywhere else the redraw is untouched. Also importable: suppress(src_img, redraw_img) -> img."""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from skimage import color as skc

FLAT_STD = 0.020            # local std of lightness (0..1) below which the source counts as flat
GRAIN = 3.0                 # max grain amplitude kept from the redraw (Lab L units)


def suppress(src, redraw):
    W, H = redraw.size
    ref = np.asarray(src.convert("RGB").resize((W, H), Image.LANCZOS), np.float32) / 255
    g = np.asarray(redraw.convert("RGB"), np.float32) / 255
    lr, lg = skc.rgb2lab(ref).astype(np.float32), skc.rgb2lab(g).astype(np.float32)
    L = cv2.GaussianBlur(lr[..., 0] / 100, (0, 0), 5.0)        # smoothed: paper grain / scan noise must not count as structure
    mu = cv2.boxFilter(L, -1, (41, 41)); var = cv2.boxFilter(L * L, -1, (41, 41)) - mu * mu
    flat = (np.sqrt(np.maximum(var, 0)) < FLAT_STD).astype(np.uint8)
    flat = cv2.erode(flat, np.ones((13, 13), np.uint8))
    flat = cv2.morphologyEx(flat, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    w = cv2.GaussianBlur(flat.astype(np.float32), (0, 0), 6.0)[..., None]
    low = cv2.GaussianBlur(lr[..., 0], (0, 0), 3.0)
    hp = lg[..., 0] - cv2.GaussianBlur(lg[..., 0], (0, 0), 3.0)
    Lf = low + GRAIN * np.tanh(hp / GRAIN)
    flat_lab = np.dstack([Lf, lr[..., 1], lr[..., 2]])
    out_lab = lg * (1 - w) + flat_lab * w
    out = skc.lab2rgb(out_lab.astype(np.float64)).clip(0, 1)
    return Image.fromarray((out * 255 + .5).astype(np.uint8)), float(flat.mean())


if __name__ == "__main__":
    res, frac = suppress(Image.open(sys.argv[1]), Image.open(sys.argv[2]))
    res.save(sys.argv[3]); print(f"flat area {100 * frac:.0f}% of the picture")
