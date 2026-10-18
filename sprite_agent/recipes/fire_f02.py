"""Recipe for fire_f02: soft incandescent fireball puff with convective swirl vortices,
filamentary flame eddies, hot ember sparks, and glowing amber falloff.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import polar_flow, ridge_noise, calibrate_alpha, calibrate_color

def make_fire_f02():
    W, H = 512, 512
    orig = fx.load("sprites/fire_f02/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Base spatial envelope
    orig_up = cv2.resize(orig, (W, H), interpolation=cv2.INTER_LANCZOS4)
    env_a = np.clip(orig_up[..., 3], 0, 1)

    r, theta, (tx, ty), (rx, ry) = polar_flow(W, H)

    # 1. Convective flow field (upward buoyancy + swirl)
    # Fire rises: dy < 0, with turbulent roll vortices
    fbm_warp1 = fx.fbm(W, H, scale=56, octaves=4, seed=12) - 0.5
    fbm_warp2 = fx.fbm(W, H, scale=40, octaves=3, seed=24) - 0.5

    warp_x = fbm_warp1 * 16.0 + tx * fbm_warp2 * 12.0
    warp_y = fbm_warp2 * 14.0 - 10.0 * (1.0 - r) # upward buoyancy draft

    # 2. Filamentary flame sheets & combustion eddies
    flame_eddies1 = ridge_noise(W, H, scale=32, octaves=4, seed=36, gamma=2.0)
    flame_eddies2 = ridge_noise(W, H, scale=16, octaves=3, seed=48, gamma=2.2)
    flame_fine = ridge_noise(W, H, scale=9, octaves=2, seed=60, gamma=2.5)

    # 3. Buoyant flame wisps / tongues
    flow_angle = -np.pi/2.0 + (fx.fbm(W, H, scale=32, octaves=2, seed=72) - 0.5) * 0.8
    flame_wisps = fx.strokes_flow(
        W, H, flow_angle,
        n=450, length=35, width=(1.2, 3.0),
        color=(1, 1, 1), alpha=(0.2, 0.5), seed=84, blur_sigma=0.8
    )[..., 3]

    # 4. Floating incandescent embers & micro sparks
    fbm_sparks = fx.fbm(W, H, scale=6, octaves=2, seed=96)
    sparks = (fbm_sparks > 0.82).astype(np.float32) * (fbm_sparks - 0.82) * 5.5

    # 5. Detail modulation
    core_mask = fx.smoothstep(0.40, 0.15, r)
    rim_mask = 1.0 - core_mask

    detail = (
        0.65
        + 0.20 * flame_eddies1
        + 0.25 * flame_eddies2 * core_mask
        + 0.15 * flame_fine
        + 0.25 * flame_wisps * rim_mask
        + 0.15 * sparks
    )

    presence = fx.smoothstep(0.02, 0.30, env_a)
    a_hd = env_a * ((1.0 - presence) + presence * detail)

    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=True, iterations=2)

    # 6. Shading: Thermal incandescence
    # Core is hot yellow-white, mid is gold, edge is glowing orange/amber
    flame_heat = np.clip(
        env_a * 0.7 + flame_eddies1 * 0.15 + flame_eddies2 * 0.15 + sparks * 0.3,
        0, 1
    )

    stops = [
        (0.00, (0.91, 0.52, 0.00)),   # Deep glowing amber/orange
        (0.35, (0.91, 0.62, 0.02)),   # Warm orange
        (0.65, (0.90, 0.74, 0.20)),   # Golden yellow
        (0.85, (0.91, 0.82, 0.42)),   # Bright yellow
        (1.00, (0.95, 0.90, 0.65))    # Hot incandescent core
    ]
    rgb_hd = fx.ramp(flame_heat, stops)
    rgb_final = calibrate_color(rgb_hd, rgb_orig, a_orig, iterations=1)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/fire_f02")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/fire_f02/draw.png")

if __name__ == "__main__":
    make_fire_f02()
