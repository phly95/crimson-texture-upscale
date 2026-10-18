"""Recipe for fire104: upward-leaping flame plume with vertical convective wisps,
filamentary combustion sheets, hot golden heart, and floating ember sparks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import ridge_noise, calibrate_alpha, calibrate_color

def make_fire104():
    W, H = 512, 512
    orig = fx.load("sprites/fire104/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Base spatial envelope
    orig_up = cv2.resize(orig, (W, H), interpolation=cv2.INTER_LANCZOS4)
    env_a = np.clip(orig_up[..., 3], 0, 1)

    # Coordinates
    x, y = fx.grid(W, H)

    # 1. Vertical convective flow warp (upward elongation + turbulent roll)
    # Fire stretches upward: stretch y, sway in x
    sway = (fx.fbm(W, H, scale=64, octaves=3, seed=14) - 0.5) * 18.0
    buoyancy = (fx.fbm(W, H, scale=48, octaves=3, seed=28) - 0.5) * 14.0 - 12.0 * (1.0 - y)

    warp_x = sway
    warp_y = buoyancy

    # 2. Vertically stretched flame filaments
    # Anisotropic scaling: scale=20 with vertical stretch
    flame1 = ridge_noise(W, H, scale=24, octaves=4, seed=35, gamma=2.0)
    flame2 = ridge_noise(W, H, scale=14, octaves=3, seed=49, gamma=2.2)
    flame_fine = ridge_noise(W, H, scale=7, octaves=2, seed=63, gamma=2.5)

    # 3. Upward licking flame strokes
    flow_angle = -np.pi/2.0 + (fx.fbm(W, H, scale=28, octaves=2, seed=77) - 0.5) * 0.5
    flame_strokes = fx.strokes_flow(
        W, H, flow_angle,
        n=400, length=45, width=(1.0, 2.5),
        color=(1, 1, 1), alpha=(0.25, 0.60), seed=88, blur_sigma=0.7
    )[..., 3]

    # 4. Floating incandescent embers & micro sparks
    fbm_sparks = fx.fbm(W, H, scale=5, octaves=2, seed=99)
    sparks = (fbm_sparks > 0.83).astype(np.float32) * (fbm_sparks - 0.83) * 6.0

    # 5. Detail modulation
    core_mask = fx.smoothstep(0.15, 0.65, env_a)

    detail = (
        0.65
        + 0.20 * flame1
        + 0.25 * flame2 * core_mask
        + 0.15 * flame_fine
        + 0.25 * flame_strokes
        + 0.15 * sparks
    )

    presence = fx.smoothstep(0.015, 0.25, env_a)
    a_hd = env_a * ((1.0 - presence) + presence * detail)

    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=True, iterations=2)

    # 6. Shading: Thermal incandescence
    flame_heat = np.clip(
        env_a * 0.70 + flame1 * 0.15 + flame2 * 0.15 + sparks * 0.35,
        0, 1
    )

    stops = [
        (0.00, (0.55, 0.12, 0.00)),   # Deep glowing amber/red edge
        (0.25, (0.72, 0.28, 0.01)),   # Warm red-orange
        (0.50, (0.85, 0.52, 0.08)),   # Fiery orange
        (0.75, (0.92, 0.75, 0.30)),   # Golden yellow
        (1.00, (0.96, 0.88, 0.58))    # Hot incandescent core
    ]
    rgb_hd = fx.ramp(flame_heat, stops)
    rgb_final = calibrate_color(rgb_hd, rgb_orig, a_orig, iterations=1)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/fire104")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/fire104/draw.png")

if __name__ == "__main__":
    make_fire104()
