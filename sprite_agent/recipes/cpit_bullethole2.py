"""Recipe for cpit_bullethole2: cockpit glass canopy bullet hole impact variant with
pulverized glass spall ring, sharp radial needle cracks, conchoidal micro-fissures,
and refractive glass caustics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import polar_flow, ridge_noise, calibrate_alpha, calibrate_color

def make_cpit_bullethole2():
    W, H = 512, 512
    orig = fx.load("sprites/cpit_bullethole2/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Base spatial envelope
    orig_up = cv2.resize(orig, (W, H), interpolation=cv2.INTER_LANCZOS4)
    env_a = np.clip(orig_up[..., 3], 0, 1)

    cx, cy = 0.511, 0.383
    r, theta, (tx, ty), (rx, ry) = polar_flow(W, H, cx=cx, cy=cy)

    # 1. Fine glass fracture network
    fissures1 = ridge_noise(W, H, scale=16, octaves=4, seed=19, gamma=3.2)
    fissures2 = ridge_noise(W, H, scale=8, octaves=3, seed=31, gamma=3.5)

    # 2. Pulverized glass grain in the spall ring around the hole
    fbm_crushed = fx.fbm(W, H, scale=5, octaves=3, seed=45)
    crushed_powder = fx.smoothstep(0.35, 0.85, fbm_crushed)

    # 3. Radial micro-cracks radiating from impact center
    field_angle = theta + (fx.fbm(W, H, scale=20, octaves=2, seed=59) - 0.5) * 0.25
    radial_cracks = fx.strokes_flow(
        W, H, field_angle,
        n=480, length=38, width=(0.7, 1.6),
        color=(1, 1, 1), alpha=(0.3, 0.7), seed=73, blur_sigma=0.4
    )[..., 3]

    # 4. Detail modulation
    core_mask = fx.smoothstep(0.22, 0.05, r)

    detail = (
        0.65
        + 0.25 * fissures1
        + 0.20 * fissures2
        + 0.30 * crushed_powder * core_mask
        + 0.25 * radial_cracks
    )

    presence = fx.smoothstep(0.015, 0.20, env_a)
    a_hd = env_a * ((1.0 - presence) + presence * detail)

    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=True, iterations=2)

    # 5. Shading: Tempered glass cyan-grey with bright specular facet edges
    specular = np.clip(
        fissures1 * 0.4 + radial_cracks * 0.4 + crushed_powder * 0.3 * core_mask,
        0, 1
    )

    stops = [
        (0.00, (0.32, 0.38, 0.38)),   # Glass edge shadow
        (0.40, (0.48, 0.55, 0.56)),   # Tempered glass body tint
        (0.70, (0.68, 0.75, 0.76)),   # Refractive facet
        (1.00, (0.87, 0.91, 0.92))    # Bright specular reflection
    ]
    rgb_hd = fx.ramp(specular, stops)
    rgb_final = calibrate_color(rgb_hd, rgb_orig, a_orig, iterations=1)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/cpit_bullethole2")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/cpit_bullethole2/draw.png")

if __name__ == "__main__":
    make_cpit_bullethole2()
