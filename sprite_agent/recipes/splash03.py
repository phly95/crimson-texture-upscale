"""Recipe for splash03: tall expanding water splash column / spout with upward-surging
water jets, turbulent breaking froth, cavitation bubbles, and airborne spray beads.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import ridge_noise, calibrate_alpha, calibrate_color

def make_splash03():
    W, H = 256, 512
    orig = fx.load("sprites/splash03/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Base spatial envelope
    orig_up = cv2.resize(orig, (W, H), interpolation=cv2.INTER_LANCZOS4)
    env_a = np.clip(orig_up[..., 3], 0, 1)

    x, y = fx.grid(W, H)

    # 1. Upward surging flow field with turbulent breaking at top
    flow_diverge = (x - 0.5) * 0.45
    flow_swirl = (fx.fbm(W, H, scale=28, octaves=3, seed=17) - 0.5) * 0.35
    flow_angle = -np.pi/2.0 + flow_diverge + flow_swirl

    # 2. Upward vertical water jet strokes
    jet_strokes = fx.strokes_flow(
        W, H, flow_angle,
        n=700, length=45, width=(0.8, 2.2),
        color=(1, 1, 1), alpha=(0.25, 0.65), seed=29, blur_sigma=0.7
    )[..., 3]

    # 3. Filamentary water ligaments & frothy webbing
    ligaments1 = ridge_noise(W, H, scale=16, octaves=4, seed=41, gamma=2.2)
    ligaments2 = ridge_noise(W, H, scale=9, octaves=3, seed=53, gamma=2.0)

    # Cellular foam bubbles in the dense core
    fbm_bubbles = fx.fbm(W, H, scale=7, octaves=3, seed=65)
    bubbles = fx.smoothstep(0.40, 0.85, fbm_bubbles)

    # Airborne droplet beads spraying off the column
    fbm_beads = fx.fbm(W, H, scale=5, octaves=2, seed=77)
    beads = (fbm_beads > 0.80).astype(np.float32) * (fbm_beads - 0.80) * 5.0

    # Billowy water puff clusters
    puffs = fx.fbm(W, H, scale=24, octaves=4, seed=89)

    # 4. Detail modulation
    core_mask = fx.smoothstep(0.3, 0.8, env_a)

    detail = (
        0.60
        + 0.20 * puffs
        + 0.25 * jet_strokes
        + 0.20 * ligaments1
        + 0.15 * ligaments2
        + 0.20 * bubbles * core_mask
        + 0.18 * beads
    )

    presence = fx.smoothstep(0.015, 0.25, env_a)
    a_hd = env_a * ((1.0 - presence) + presence * detail)

    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=True, iterations=2)

    # 5. Shading: Smooth interpolation from dark edge tint to bright foam core
    core_intensity = np.clip(
        env_a * 0.7 + jet_strokes * 0.15 + bubbles * 0.15 + beads * 0.3,
        0, 1
    )

    stops = [
        (0.00, (0.24, 0.25, 0.25)),   # Shadowed edge
        (0.35, (0.45, 0.46, 0.47)),   # Midtone grey
        (0.65, (0.75, 0.76, 0.77)),   # Light foam
        (1.00, (0.91, 0.92, 0.93))    # Pure white foam core
    ]
    rgb_hd = fx.ramp(core_intensity, stops)
    rgb_final = calibrate_color(rgb_hd, rgb_orig, a_orig, iterations=1)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/splash03")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/splash03/draw.png")

if __name__ == "__main__":
    make_splash03()
