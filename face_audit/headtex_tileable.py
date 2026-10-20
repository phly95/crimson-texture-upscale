"""headtex_tileable.py IN OUT [AXES]: make a texture wrap seamlessly along x (and/or y): near each edge cross-fade to the half-period-shifted copy (only the outer ~15%)"""
import sys
import numpy as np
from PIL import Image
def periodic(img, axes="x", frac=0.15):
    out = img.astype(np.float32)
    for ax in axes:
        a = 1 if ax == "x" else 0; n = out.shape[a]; t = np.arange(n) / (n - 1); e = np.minimum(t, 1 - t) / frac
        w = np.clip(e, 0, 1); w = w * w * (3 - 2 * w); w = w.reshape((1, n, 1) if a == 1 else (n, 1, 1))
        out = out * w + np.roll(out, n // 2, axis=a) * (1 - w)
    return np.clip(out + .5, 0, 255).astype(np.uint8)
def seam_err(img, ax):
    x = img.astype(np.float32); return float(np.abs(x[:, 0] - x[:, -1]).mean()) if ax == "x" else float(np.abs(x[0] - x[-1]).mean())
if __name__ == "__main__":
    im = np.array(Image.open(sys.argv[1]).convert("RGB")); axes = sys.argv[3] if len(sys.argv) > 3 else "x"; o = periodic(im, axes)
    print("edge mismatch before x %.1f y %.1f after x %.1f y %.1f" % (seam_err(im, "x"), seam_err(im, "y"), seam_err(o, "x"), seam_err(o, "y"))); Image.fromarray(o).save(sys.argv[2])
