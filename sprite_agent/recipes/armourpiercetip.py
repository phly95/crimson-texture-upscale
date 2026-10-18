"""Recipe for armourpiercetip: incandescent armor-piercing projectile impact flash,
with white-hot plasma core, starburst optical flare prongs, radial spark streamers,
and deep crimson optical bloom.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import polar_flow, ridge_noise, calibrate_alpha, calibrate_color

def make_armourpiercetip():
    W, H = 256, 256
    orig = fx.load("sprites/armourpiercetip/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Base spatial envelope
    orig_up = cv2.resize(orig, (W, H), interpolation=cv2.INTER_LANCZOS4)
    env_a = np.clip(orig_up[..., 3], 0, 1)

    cx, cy = 0.484, 0.484
    r, theta, (tx, ty), (rx, ry) = polar_flow(W, H, cx=cx, cy=cy)

    # 1. Optical starburst flare prongs (4-point primary diamond + 4-point secondary)
    flare_primary = (np.cos(4.0 * (theta - 0.15)) * 0.5 + 0.5) ** 6.0
    flare_secondary = (np.cos(4.0 * (theta - 0.15 + np.pi/4)) * 0.5 + 0.5) ** 10.0
    flare_spikes = flare_primary * 0.35 + flare_secondary * 0.15

    # Flare falloff with distance
    flare_band = flare_spikes * fx.smoothstep(0.25, 0.04, r)

    # 2. Plasma core turbulence & filamentary arcs
    fbm_plasma = fx.fbm(W, H, scale=24, octaves=4, seed=101)
    plasma_filaments = ridge_noise(W, H, scale=12, octaves=3, seed=202, gamma=2.2)

    # 3. Fine radial energy streamers / spark trails
    field_angle = theta + (fx.fbm(W, H, scale=16, octaves=2, seed=303) - 0.5) * 0.3
    streamers = fx.strokes_flow(
        W, H, field_angle,
        n=250, length=25, width=(0.8, 1.8),
        color=(1, 1, 1), alpha=(0.3, 0.7), seed=404, blur_sigma=0.5
    )[..., 3]

    # 4. Micro sparks & incandescent beads
    fbm_sparks = fx.fbm(W, H, scale=5, octaves=2, seed=505)
    sparks = (fbm_sparks > 0.85).astype(np.float32) * (fbm_sparks - 0.85) * 6.0

    # 5. Detail modulation
    core_mask = fx.smoothstep(0.12, 0.03, r)
    mid_mask = fx.smoothstep(0.30, 0.10, r) * (1.0 - core_mask)

    detail = (
        0.75
        + 0.25 * flare_band
        + 0.20 * plasma_filaments * core_mask
        + 0.25 * streamers * mid_mask
        + 0.20 * sparks * mid_mask
    )

    presence = fx.smoothstep(0.015, 0.20, env_a)
    a_hd = env_a * ((1.0 - presence) + presence * detail)

    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=True, iterations=2)

    # 6. Shading: Blackbody incandescence ramp
    # Core is white-hot/pale yellow, mid is orange/amber, halo is deep crimson
    # Temperature profile driven by original brightness + flare prongs
    temp = np.clip(env_a ** 1.1 + flare_band * 0.4 + plasma_filaments * 0.15 * core_mask, 0, 1)

    # Blackbody color stops:
    stops = [
        (0.00, (1.0, 0.02, 0.00)),   # Deep crimson edge
        (0.20, (1.0, 0.08, 0.01)),   # Hot red halo
        (0.45, (1.0, 0.28, 0.05)),   # Orange
        (0.65, (1.0, 0.55, 0.15)),   # Amber / gold
        (0.85, (1.0, 0.88, 0.55)),   # Incandescent yellow
        (1.00, (1.0, 0.98, 0.92))    # White-hot plasma core
    ]
    rgb_hd = fx.ramp(temp, stops)

    rgb_final = calibrate_color(rgb_hd, rgb_orig, a_orig, iterations=1)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/armourpiercetip")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/armourpiercetip/draw.png")

if __name__ == "__main__":
    make_armourpiercetip()
