"""Recipe for srf0014: vertically tiling shoreline surf / wave breaker strip variant with
dense frothy foam lacing, cellular aeration bubbles, and translucent wash falloff.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import vtile_resize, ridge_noise, calibrate_alpha, calibrate_color

def make_srf0014():
    W, H = 256, 512
    orig = fx.load("sprites/srf0014/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Base spatial envelope with vertical periodic wrapping
    env_a = vtile_resize(a_orig, (W, H), cv2.INTER_CUBIC)

    # 1. Multi-scale foam webbing & breaker ridges (tile=True for vertical wrapping)
    foam1 = ridge_noise(W, H, scale=20, octaves=4, seed=18, gamma=2.2, tile=True)
    foam2 = ridge_noise(W, H, scale=10, octaves=3, seed=32, gamma=2.0, tile=True)
    foam_fine = ridge_noise(W, H, scale=5, octaves=2, seed=44, gamma=2.5, tile=True)

    # 2. Aerated bubble clusters & froth beads
    fbm_bubbles = fx.fbm(W, H, scale=7, octaves=3, seed=58, tile=True)
    bubbles = fx.smoothstep(0.40, 0.85, fbm_bubbles)

    # 3. Turbulent breaker wash lines
    warp_dx = (fx.fbm(W, H, scale=32, octaves=3, seed=70, tile=True) - 0.5) * 10.0
    warp_dy = (fx.fbm(W, H, scale=32, octaves=3, seed=82, tile=True) - 0.5) * 6.0

    breaker_foam = fx.warp(foam1 * 0.6 + foam2 * 0.4, warp_dx, warp_dy)

    # 4. Detail modulation
    detail = (
        0.70
        + 0.25 * breaker_foam
        + 0.15 * foam_fine
        + 0.15 * bubbles
    )

    presence = fx.smoothstep(0.015, 0.25, env_a)
    a_hd = env_a * ((1.0 - presence) + presence * detail)

    # Calibrate alpha with vtile=True to guarantee seamless vertical tiling
    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=False, iterations=2, vtile=True)

    # 5. Shading: Neutral seafoam grey with bright white foam highlights
    foam_highlight = np.clip(
        env_a * 0.6 + breaker_foam * 0.3 + bubbles * 0.2,
        0, 1
    )

    stops = [
        (0.00, (0.42, 0.42, 0.42)),   # Deeper water shadow
        (0.40, (0.58, 0.58, 0.58)),   # Neutral seafoam grey
        (0.75, (0.78, 0.78, 0.78)),   # Bright froth
        (1.00, (0.92, 0.92, 0.93))    # Pure white foam crest
    ]
    rgb_hd = fx.ramp(foam_highlight, stops)
    rgb_final = calibrate_color(rgb_hd, rgb_orig, a_orig, iterations=1)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/srf0014")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/srf0014/draw.png")

if __name__ == "__main__":
    make_srf0014()
