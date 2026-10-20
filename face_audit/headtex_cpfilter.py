"""headtex_cpfilter.py SRC OUT [levels ink]: geometry-safe cartoonisation of an already-aligned baked UV texture: mean-shift flattening, soft lightness posterise, thin ink only on strong, broad edges.
The layout cannot change (no generation involved)."""
import sys
import cv2, numpy as np
from PIL import Image
import skimage.color as skc
def toon(rgb, levels=7, ink=0.5, sp=9, sr=22):
    x = cv2.pyrMeanShiftFiltering(rgb, sp, sr, maxLevel=2)
    x = cv2.bilateralFilter(x, 9, 30, 7)
    lab = skc.rgb2lab(x / 255.).astype(np.float32); L = lab[..., 0]
    q = np.round(L / 100 * levels) / levels * 100; w = 0.55                                             # soft posterise: blend toward the quantised lightness
    lab[..., 0] = L * (1 - w) + q * w
    lab[..., 1:] *= 1.12                                                                                     # slightly richer colour
    out = (np.clip(skc.lab2rgb(lab.astype(np.float64)), 0, 1) * 255).astype(np.uint8)
    if ink > 0:
        g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (0, 0), 2.0).astype(np.float32)
        e = cv2.GaussianBlur(g, (0, 0), 1.5) - cv2.GaussianBlur(g, (0, 0), 4.0)                              # difference of Gaussians
        m = np.clip((np.abs(e) - 3.0) / 6.0, 0, 1) * ink; out = (out * (1 - m[..., None] * 0.8)).astype(np.uint8)
    return out
if __name__ == "__main__":
    s = np.array(Image.open(sys.argv[1]).convert("RGB")); lv = int(sys.argv[3]) if len(sys.argv) > 3 else 7; ik = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    Image.fromarray(toon(s, lv, ik)).save(sys.argv[2])
