"""Recipe for fire_f04: cooling fireball puff transitioning from dense crimson combustion
core to turbulent golden flame tongues, burnt mahogany smoke billows, and ember sparks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import polar_flow, ridge_noise, calibrate_alpha, calibrate_color

def make_fire_f04():
    W, H = 512, 512
    orig = fx.load("sprites/fire_f04/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Base spatial envelope
    orig_up = cv2.resize(orig, (W, H), interpolation=cv2.INTER_LANCZOS4)
    env_a = np.clip(orig_up[..., 3], 0, 1)

    r, theta, (tx, ty), (rx, ry) = polar_flow(W, H)

    # 1. Convective flow field (upward buoyancy + swirling eddies)
    fbm_warp1 = fx.fbm(W, H, scale=50, octaves=4, seed=17) - 0.5
    fbm_warp2 = fx.fbm(W, H, scale=36, octaves=3, seed=29) - 0.5

    warp_x = fbm_warp1 * 18.0 + tx * fbm_warp2 * 14.0
    warp_y = fbm_warp2 * 16.0 - 12.0 * (1.0 - r) # upward buoyancy draft

    # 2. Filamentary flame sheets & combustion eddies
    flame_eddies1 = ridge_noise(W, H, scale=30, octaves=4, seed=41, gamma=2.0)
    flame_eddies2 = ridge_noise(W, H, scale=15, octaves=3, seed=53, gamma=2.2)
    flame_fine = ridge_noise(W, H, scale=8, octaves=2, seed=65, gamma=2.5)

    # 3. Buoyant flame wisps / smoke tendrils
    flow_angle = -np.pi/2.0 + (fx.fbm(W, H, scale=32, octaves=2, seed=77) - 0.5) * 0.9
    flame_wisps = fx.strokes_flow(
        W, H, flow_angle,
        n=480, length=38, width=(1.0, 2.8),
        color=(1, 1, 1), alpha=(0.2, 0.55), seed=89, blur_sigma=0.8
    )[..., 3]

    # 4. Floating incandescent embers & micro sparks
    fbm_sparks = fx.fbm(W, H, scale=6, octaves=2, seed=93)
    sparks = (fbm_sparks > 0.82).astype(np.float32) * (fbm_sparks - 0.82) * 5.5

    # 5. Detail modulation
    core_mask = fx.smoothstep(0.42, 0.15, r)
    rim_mask = 1.0 - core_mask

    detail = (
        0.65
        + 0.20 * flame_eddies1
        + 0.25 * flame_eddies2 * core_mask
        + 0.15 * flame_fine
        + 0.25 * flame_wisps * rim_mask
        + 0.18 * sparks
    )

    presence = fx.smoothstep(0.02, 0.30, env_a)
    a_hd = env_a * ((1.0 - presence) + presence * detail)

    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=True, iterations=2)

    # 6. Shading: Cooling combustion palette
    # Low temp = dark mahogany smoke, mid = deep red core, high = fiery golden orange
    flame_heat = np.clip(
        env_a * 0.65 + flame_eddies1 * 0.20 + flame_eddies2 * 0.15 + sparks * 0.35,
        0, 1
    )

    stops = [
        (0.00, (0.35, 0.01, 0.00)),   # Dark burnt mahogany smoke
        (0.25, (0.60, 0.04, 0.01)),   # Deep crimson smoke
        (0.50, (0.78, 0.18, 0.01)),   # Fiery red-orange
        (0.75, (0.88, 0.38, 0.02)),   # Bright orange
        (1.00, (0.95, 0.58, 0.05))    # Golden flame crest
    ]
    rgb_hd = fx.ramp(flame_heat, stops)
    rgb_final = calibrate_color(rgb_hd, rgb_orig, a_orig, iterations=1)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/fire_f04")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/fire_f04/draw.png")

if __name__ == "__main__":
    make_fire_f04()
