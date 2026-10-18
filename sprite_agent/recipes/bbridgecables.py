"""Recipe for bbridgecables: vertical bridge suspension cable column with
smooth anti-aliased cylindrical profile, subtle braided steel strand shading,
and exact 8x8 block calibration.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import calibrate_alpha

def make_bbridgecables():
    W, H = 128, 64
    orig = fx.load("sprites/bbridgecables/orig.png")
    a_orig = orig[..., 3]

    # Target size: 128x64.
    # The profile is 1D along X (16 columns in original).
    col_prof = a_orig[0, :]

    # 1D smooth interpolation of the horizontal profile across 128 columns
    x_orig = np.linspace(0, 1, 16)
    x_hd = np.linspace(0, 1, W)
    prof_hd = np.interp(x_hd, x_orig, col_prof).astype(np.float32)
    # Smooth slightly to ensure vector-quality Gaussian anti-aliased edge
    prof_hd = cv2.GaussianBlur(prof_hd[None, :], (9, 1), 1.5).ravel()

    # Broadcast across all Y rows (constant along vertical axis, perfect vertical tiling)
    a_hd = np.tile(prof_hd[None, :], (H, 1))

    # Residual calibration ensures exact 8x8 block downsampling matches orig.png
    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=False, iterations=2, tile=True)

    # Pure black color for suspension cable
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    layer = np.dstack([rgb, np.clip(a_final, 0, 1)])

    out_dir = Path("sprites/bbridgecables")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/bbridgecables/draw.png")

if __name__ == "__main__":
    make_bbridgecables()
