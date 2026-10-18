"""Recipe for mptrailer_shadow: crisp mobile trailer / vehicle shadow
with smooth clean contours and subtle anti-aliased edge falloff.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import calibrate_alpha

def make_mptrailer_shadow():
    W, H = 256, 512
    orig = fx.load("sprites/mptrailer_shadow/orig.png")
    a_orig = orig[..., 3]

    # Smooth continuous upscaling of the shadow silhouette using Lanczos
    orig_up = cv2.resize(orig, (W, H), interpolation=cv2.INTER_LANCZOS4)
    a_hd = np.clip(orig_up[..., 3], 0, 1)

    # Very gentle blur for clean anti-aliased contours
    a_hd = cv2.GaussianBlur(a_hd, (0, 0), 1.2)

    # Residual calibration ensures exact 8x8 block downsampling matches orig.png
    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=True, iterations=2)

    # Pure black color for shadow
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    layer = np.dstack([rgb, np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/mptrailer_shadow")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/mptrailer_shadow/draw.png")

if __name__ == "__main__":
    make_mptrailer_shadow()
