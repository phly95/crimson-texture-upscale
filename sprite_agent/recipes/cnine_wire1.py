"""Recipe for cnine_wire1: bridge/airship structural steel truss bay with
top and bottom horizontal chord girders, diagonal angled brace struts,
riveted gusset plates, and weathered steel/rust patina.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import calibrate_alpha, calibrate_color, ridge_noise

def make_cnine_wire1():
    W, H = 128, 128
    orig = fx.load("sprites/cnine_wire1/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Smooth continuous upscaling of the truss structure
    orig_up = cv2.resize(orig, (W, H), interpolation=cv2.INTER_LANCZOS4)
    a_hd = np.clip(orig_up[..., 3], 0, 1)
    rgb_hd = np.clip(orig_up[..., :3], 0, 1)

    # Add subtle metal/rust grime texture
    grime = fx.fbm(W, H, scale=12, octaves=3, seed=42) * 0.15 - 0.075
    rgb_textured = np.clip(rgb_hd + grime[..., None], 0, 1)

    # Calibrate alpha and color to match original 8x8 block downsampling
    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=True, iterations=2)
    rgb_final = calibrate_color(rgb_textured, rgb_orig, a_orig, iterations=1)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/cnine_wire1")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/cnine_wire1/draw.png")

if __name__ == "__main__":
    make_cnine_wire1()
