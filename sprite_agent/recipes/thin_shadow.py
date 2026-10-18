"""Recipe for thin_shadow: clean horizontal shadow band with perfectly smooth
top and bottom Gaussian falloff and constant horizontal profile.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import calibrate_alpha

def make_thin_shadow():
    W, H = 128, 128
    orig = fx.load("sprites/thin_shadow/orig.png")
    a_orig = orig[..., 3]

    # Target size: 128x128.
    # The profile is 1D along Y (rows 0 to 15 in original).
    # Smooth 1D interpolation of the original row profile
    row_prof = a_orig[:, 0]

    # Upsample the 16 profile points to 128 points using cubic spline/resampling
    y_orig = np.linspace(0, 1, 16)
    y_hd = np.linspace(0, 1, H)
    prof_hd = np.interp(y_hd, y_orig, row_prof).astype(np.float32)
    # Smooth with a gentle 1D Gaussian to remove linear interpolation facets
    prof_hd = cv2.GaussianBlur(prof_hd[:, None], (1, 9), 2.0).ravel()

    # Broadcast across all X columns (constant along X)
    a_hd = np.tile(prof_hd[:, None], (1, W))

    # Residual calibration ensures exact 8x8 block average matches the original
    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=False, iterations=2, tile=True)

    # Clean black color for shadow
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    layer = np.dstack([rgb, np.clip(a_final, 0, 1)])

    out_dir = Path("sprites/thin_shadow")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/thin_shadow/draw.png")

if __name__ == "__main__":
    make_thin_shadow()
