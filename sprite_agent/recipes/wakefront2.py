"""Recipe for wavefront2: horizontally tiling boat/plane water wake front strip with
dense frothy keel wash, cellular aeration bubbles, and wispy water plume feathering.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import htile_resize, ridge_noise, calibrate_alpha, calibrate_color

def make_wakefront2():
    W, H = 512, 256
    orig = fx.load("sprites/wakefront2/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Base spatial envelope with horizontal periodic wrapping
    env_a = htile_resize(a_orig, (W, H), cv2.INTER_CUBIC)

    # 1. Multi-scale wake streaks & foam filaments (tile=True for horizontal wrapping)
    wake1 = ridge_noise(W, H, scale=24, octaves=4, seed=14, gamma=2.2, tile=True)
    wake2 = ridge_noise(W, H, scale=12, octaves=3, seed=28, gamma=2.0, tile=True)
    wake_fine = ridge_noise(W, H, scale=6, octaves=2, seed=42, gamma=2.5, tile=True)

    # 2. Aerated foam bubbles & froth beads in the dense keel wash
    fbm_bubbles = fx.fbm(W, H, scale=8, octaves=3, seed=56, tile=True)
    bubbles = fx.smoothstep(0.40, 0.85, fbm_bubbles)

    # 3. Turbulent diagonal shear warp (water churning off the hull)
    warp_dx = (fx.fbm(W, H, scale=32, octaves=3, seed=70, tile=True) - 0.5) * 12.0
    warp_dy = (fx.fbm(W, H, scale=32, octaves=3, seed=84, tile=True) - 0.5) * 6.0

    wake_foam = fx.warp(wake1 * 0.6 + wake2 * 0.4, warp_dx, warp_dy)

    # 4. Detail modulation
    detail = (
        0.70
        + 0.25 * wake_foam
        + 0.15 * wake_fine
        + 0.15 * bubbles
    )

    presence = fx.smoothstep(0.015, 0.25, env_a)
    a_hd = env_a * ((1.0 - presence) + presence * detail)

    # Calibrate alpha with htile=True to guarantee seamless horizontal tiling
    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=False, iterations=2, htile=True)

    # 5. Shading: Neutral seafoam grey with bright white foam highlights
    foam_highlight = np.clip(
        env_a * 0.6 + wake_foam * 0.3 + bubbles * 0.2,
        0, 1
    )

    stops = [
        (0.00, (0.44, 0.44, 0.44)),   # Deeper water shadow
        (0.40, (0.58, 0.58, 0.58)),   # Neutral seafoam grey
        (0.75, (0.75, 0.75, 0.75)),   # Bright froth
        (1.00, (0.90, 0.90, 0.91))    # Pure white foam crest
    ]
    rgb_hd = fx.ramp(foam_highlight, stops)
    rgb_final = calibrate_color(rgb_hd, rgb_orig, a_orig, iterations=1)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/wakefront2")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/wakefront2/draw.png")

if __name__ == "__main__":
    make_wakefront2()
