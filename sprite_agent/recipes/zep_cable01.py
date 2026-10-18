"""Recipe for zep_cable01: zeppelin mooring / suspension cable strip with
smooth anti-aliased cylindrical profile and seamless horizontal tiling.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import calibrate_alpha

def make_zep_cable01():
    W, H = 128, 128
    orig = fx.load("sprites/zep_cable01/orig.png")
    a_orig = orig[..., 3]

    # Target size: 128x128. Profile is 1D along Y (16 rows in original).
    row_prof = a_orig[:, 0]

    # 1D smooth interpolation of the row profile across 128 rows
    y_orig = np.linspace(0, 1, 16)
    y_hd = np.linspace(0, 1, H)
    prof_hd = np.interp(y_hd, y_orig, row_prof).astype(np.float32)
    prof_hd = cv2.GaussianBlur(prof_hd[:, None], (1, 9), 1.5).ravel()

    # Broadcast across all X columns (constant along X, perfect horizontal tiling)
    a_hd = np.tile(prof_hd[:, None], (1, W))

    # Residual calibration ensures exact 8x8 block downsampling matches orig.png
    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=False, iterations=2, tile=True)

    # Pure black color for mooring cable
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    layer = np.dstack([rgb, np.clip(a_final, 0, 1)])

    out_dir = Path("sprites/zep_cable01")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/zep_cable01/draw.png")

if __name__ == "__main__":
    make_zep_cable01()
