"""Recipe for grate01: perforated heavy steel drainage/ventilation grate plate
with 3x3 array of rounded square cutouts, machined chamfered bevels,
subtle surface scuffs/wear, and seamless 2D tiling.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import calibrate_alpha, calibrate_color, periodic_resize

def make_grate01():
    W, H = 256, 256
    orig = fx.load("sprites/grate01/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Smooth continuous upscaling with 2D periodic wrapping
    env_a = periodic_resize(a_orig, (W, H), cv2.INTER_CUBIC)
    rgb_hd = periodic_resize(rgb_orig, (W, H), cv2.INTER_CUBIC)

    # Add subtle metal scratches / scuffs
    scuffs = fx.fbm(W, H, scale=12, octaves=4, seed=44, tile=True) * 0.08 - 0.04
    rgb_textured = np.clip(rgb_hd + scuffs[..., None], 0, 1)

    # Calibrate alpha and color with tile=True for seamless 2D tiling
    a_final = calibrate_alpha(env_a, a_orig, strict_zero=False, iterations=2, tile=True)
    rgb_final = calibrate_color(rgb_textured, rgb_orig, a_orig, iterations=1, tile=True)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/grate01")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/grate01/draw.png")

if __name__ == "__main__":
    make_grate01()
