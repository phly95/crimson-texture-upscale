"""Recipe for firetruck_shadow: clean vehicle shadow silhouette with smooth
curved contours, window cutout, soft diffuse shadow falloff on roof/hood,
and solid shadow undercarriage.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import calibrate_alpha

def make_firetruck_shadow():
    W, H = 1024, 512
    orig = fx.load("sprites/firetruck_shadow/orig.png")
    a_orig = orig[..., 3]

    # Smooth continuous upscaling of the shadow silhouette
    # Using Lanczos gives a high-quality smooth anti-aliased reconstruction of the shadow shape
    orig_up = cv2.resize(orig, (W, H), interpolation=cv2.INTER_LANCZOS4)
    a_hd = np.clip(orig_up[..., 3], 0, 1)

    # Gentle smoothing to ensure clean vector-like gradient falloff
    a_hd = cv2.GaussianBlur(a_hd, (0, 0), 1.5)

    # Residual calibration ensures exact 8x8 block downsampling matches orig.png
    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=True, iterations=2)

    # Shading: orig color is deep midnight black/blue [0.0, 0.0314, 0.0627]
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., 0] = 0.0
    rgb[..., 1] = 8.0 / 255.0
    rgb[..., 2] = 16.0 / 255.0

    layer = np.dstack([rgb, np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/firetruck_shadow")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/firetruck_shadow/draw.png")

if __name__ == "__main__":
    make_firetruck_shadow()
