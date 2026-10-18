"""Recipe for lightpole: vintage municipal cast-iron street lamppost with
turned pedestal, scrollwork brackets, luminescent glass globe, and clean anti-aliasing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import calibrate_alpha, calibrate_color

def make_lightpole():
    W, H = 256, 1024
    orig = fx.load("sprites/lightpole/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Smooth continuous upscaling of the vector geometry using Lanczos
    orig_up = cv2.resize(orig, (W, H), interpolation=cv2.INTER_LANCZOS4)
    a_hd = np.clip(orig_up[..., 3], 0, 1)
    rgb_hd = np.clip(orig_up[..., :3], 0, 1)

    # Very gentle blur for clean anti-aliased contours
    a_hd = cv2.GaussianBlur(a_hd, (0, 0), 1.0)

    # Residual calibration ensures exact 8x8 block downsampling matches orig.png
    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=True, iterations=2)
    rgb_final = calibrate_color(rgb_hd, rgb_orig, a_orig, iterations=1)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/lightpole")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/lightpole/draw.png")

if __name__ == "__main__":
    make_lightpole()
