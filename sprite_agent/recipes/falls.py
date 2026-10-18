"""Recipe for falls: seamless vertically cascading waterfall texture with
shearing water curtain filaments, aerated foam striations, and turbulent spray.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import periodic_resize, ridge_noise, calibrate_alpha

def make_falls():
    W, H = 256, 256
    orig = fx.load("sprites/falls/orig.png")
    a_orig = orig[..., 3]

    # Base spatial envelope with periodic wrapping
    env_a = periodic_resize(a_orig, (W, H), cv2.INTER_CUBIC)

    # 1. Multi-scale vertical waterfall filaments (all tile=True)
    # Primary vertical cascading curtain streams
    streams1 = ridge_noise(W, H, scale=16, octaves=4, seed=12, gamma=2.0, tile=True)
    streams2 = ridge_noise(W, H, scale=8, octaves=3, seed=24, gamma=2.2, tile=True)
    streams_fine = ridge_noise(W, H, scale=4, octaves=2, seed=36, gamma=2.5, tile=True)

    # 2. Aerated foam bubbles & droplet texture carried down the falls
    fbm_bubbles = fx.fbm(W, H, scale=6, octaves=3, seed=48, tile=True)
    bubbles = fx.smoothstep(0.40, 0.85, fbm_bubbles)

    # 3. Horizontal turbulent shearing warp across the vertical streams
    warp_dx = (fx.fbm(W, H, scale=32, octaves=3, seed=60, tile=True) - 0.5) * 8.0
    warp_dy = (fx.fbm(W, H, scale=32, octaves=3, seed=72, tile=True) - 0.5) * 4.0

    streams_warped = fx.warp(streams1 * 0.6 + streams2 * 0.4, warp_dx, warp_dy)

    # 4. Detail modulation
    detail = (
        0.75
        + 0.25 * streams_warped
        + 0.15 * streams_fine
        + 0.15 * bubbles
    )

    a_hd = env_a * detail

    # Calibrate alpha with tile=True to preserve seamless wrapping
    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=False, iterations=2, tile=True)

    # 5. Shading: Waterfall is pure white with alpha modulating thickness/aeration
    rgb_final = np.ones((H, W, 3), dtype=np.float32)

    layer = np.dstack([rgb_final, np.clip(a_final, 0, 1)])

    out_dir = Path("sprites/falls")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/falls/draw.png")

if __name__ == "__main__":
    make_falls()
