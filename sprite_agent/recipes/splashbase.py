"""Recipe for splashbase: billowy water splash base with outward radial spray,
dense aerated foam core, bubbly cavitation, and airborne droplet beads.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import polar_flow, ridge_noise, calibrate_alpha, calibrate_color

def make_splashbase():
    W, H = 512, 512
    orig = fx.load("sprites/splashbase/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # 1. Base spatial envelope
    orig_up = cv2.resize(orig, (W, H), interpolation=cv2.INTER_LANCZOS4)
    env_a = np.clip(orig_up[..., 3], 0, 1)

    # Coordinates & flow
    r, theta, (tx, ty), (rx, ry) = polar_flow(W, H)

    # 2. Multi-scale turbulent billowing
    # Domain warp pushing outward along radial rays
    warp_amp = fx.fbm(W, H, scale=64, octaves=4, seed=11) - 0.5
    warp_swirl = fx.fbm(W, H, scale=48, octaves=3, seed=22) - 0.5
    warp_x = warp_amp * rx * 18.0 + warp_swirl * tx * 10.0
    warp_y = warp_amp * ry * 18.0 + warp_swirl * ty * 10.0

    # 3. Radial spray streaks (water flung violently outward)
    field_angle = theta + (fx.fbm(W, H, scale=32, octaves=2, seed=33) - 0.5) * 0.4
    spray_strokes = fx.strokes_flow(
        W, H, field_angle,
        n=600, length=45, width=(1.0, 2.5),
        color=(1, 1, 1), alpha=(0.25, 0.65), seed=44, blur_sigma=0.8
    )[..., 3]

    # 4. Dense aerated foam core (bubble cavitation & webbing)
    foam1 = ridge_noise(W, H, scale=18, octaves=4, seed=55, gamma=2.2)
    foam2 = ridge_noise(W, H, scale=10, octaves=3, seed=66, gamma=2.0)

    # Bubbly cellular pores
    fbm_bubbles = fx.fbm(W, H, scale=8, octaves=3, seed=77)
    bubbles = fx.smoothstep(0.40, 0.85, fbm_bubbles)

    # Droplet beads flying in the spray
    fbm_droplets = fx.fbm(W, H, scale=5, octaves=2, seed=88)
    droplets = (fbm_droplets > 0.80).astype(np.float32) * (fbm_droplets - 0.80) * 5.0

    # Billowy water puff clusters
    puffs = fx.fbm(W, H, scale=28, octaves=4, seed=99)

    # Detail modulation
    core_mask = fx.smoothstep(0.40, 0.15, r)
    rim_mask = 1.0 - core_mask

    detail = (
        0.55
        + 0.25 * puffs
        + 0.25 * foam1 * core_mask
        + 0.20 * foam2
        + 0.20 * bubbles * core_mask
        + 0.35 * spray_strokes * rim_mask
        + 0.20 * droplets
    )

    presence = fx.smoothstep(0.02, 0.25, env_a)
    a_hd = env_a * ((1.0 - presence) + presence * detail)

    # Calibrate alpha to match Image.BOX downsampling
    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=True, iterations=2)

    # 5. Shading & Color
    # Foam highlights are bright pure white; troughs are pale water mist
    highlight_intensity = np.clip(
        puffs * 0.3 + foam1 * 0.3 + bubbles * 0.2 + droplets * 0.4 + spray_strokes * 0.3,
        0, 1
    )

    base_col = np.array([0.85, 0.85, 0.86], dtype=np.float32)
    white_col = np.array([0.96, 0.96, 0.97], dtype=np.float32)
    shade_col = np.array([0.76, 0.76, 0.77], dtype=np.float32)

    rgb_hd = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        rgb_hd[..., c] = base_col[c] + (white_col[c] - base_col[c]) * highlight_intensity * 0.7 \
                                     - (base_col[c] - shade_col[c]) * (1.0 - highlight_intensity) * 0.3

    rgb_final = calibrate_color(rgb_hd, rgb_orig, a_orig, iterations=1)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/splashbase")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/splashbase/draw.png")

if __name__ == "__main__":
    make_splashbase()
