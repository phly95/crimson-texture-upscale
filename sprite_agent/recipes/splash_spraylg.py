"""Recipe for splash_spraylg: large directional water spray plume with upward-right
flung spray tendrils, Rayleigh-Plateau droplet ligaments, fine mist, and airborne beads.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import ridge_noise, calibrate_alpha, calibrate_color

def make_splash_spraylg():
    W, H = 512, 512
    orig = fx.load("sprites/splash_spraylg/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Base spatial envelope
    orig_up = cv2.resize(orig, (W, H), interpolation=cv2.INTER_LANCZOS4)
    env_a = np.clip(orig_up[..., 3], 0, 1)

    x, y = fx.grid(W, H)

    # 1. Directional spray flow field: emerging from bottom-left, spreading upward and right
    # Origin approximately at (0.1, 0.9)
    dx = x - 0.1
    dy = y - 0.9
    base_flow_angle = np.arctan2(dy, dx) # angles spread outward from origin
    # Add turbulent swirl to flow
    swirl = (fx.fbm(W, H, scale=36, octaves=3, seed=15) - 0.5) * 0.45
    spray_angle = base_flow_angle + swirl

    # 2. Fine directional spray strokes (water jets & droplets)
    spray_strokes = fx.strokes_flow(
        W, H, spray_angle,
        n=750, length=50, width=(0.8, 2.2),
        color=(1, 1, 1), alpha=(0.25, 0.65), seed=25, blur_sigma=0.7
    )[..., 3]

    # Secondary ultra-fine wispy mist
    spray_mist = fx.strokes_flow(
        W, H, spray_angle,
        n=500, length=25, width=(0.5, 1.2),
        color=(1, 1, 1), alpha=(0.2, 0.45), seed=35, blur_sigma=0.5
    )[..., 3]

    # 3. Filamentary water ligaments & droplet beads
    ligaments1 = ridge_noise(W, H, scale=18, octaves=4, seed=45, gamma=2.2)
    ligaments2 = ridge_noise(W, H, scale=10, octaves=3, seed=55, gamma=2.0)

    # Micro droplet beads
    fbm_beads = fx.fbm(W, H, scale=5, octaves=2, seed=65)
    beads = (fbm_beads > 0.78).astype(np.float32) * (fbm_beads - 0.78) * 5.0

    # Billowy mist clouds
    mist_clouds = fx.fbm(W, H, scale=32, octaves=3, seed=75)

    # 4. Detail modulation
    detail = (
        0.55
        + 0.20 * mist_clouds
        + 0.30 * spray_strokes
        + 0.15 * spray_mist
        + 0.20 * ligaments1
        + 0.15 * ligaments2
        + 0.18 * beads
    )

    presence = fx.smoothstep(0.015, 0.20, env_a)
    a_hd = env_a * ((1.0 - presence) + presence * detail)

    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=True, iterations=2)

    # 5. Shading: Pale sky/seafoam tint with bright droplet specular crests
    crest_intensity = np.clip(
        spray_strokes * 0.35 + ligaments1 * 0.25 + beads * 0.45,
        0, 1
    )

    base_color = np.array([0.785, 0.793, 0.812], dtype=np.float32)
    highlight_color = np.array([0.88, 0.895, 0.92], dtype=np.float32)
    trough_color = np.array([0.74, 0.75, 0.78], dtype=np.float32)

    rgb_hd = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        rgb_hd[..., c] = base_color[c] + (highlight_color[c] - base_color[c]) * crest_intensity * 0.65 \
                                       - (base_color[c] - trough_color[c]) * (1.0 - crest_intensity) * 0.25

    rgb_final = calibrate_color(rgb_hd, rgb_orig, a_orig, iterations=1)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/splash_spraylg")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/splash_spraylg/draw.png")

if __name__ == "__main__":
    make_splash_spraylg()
