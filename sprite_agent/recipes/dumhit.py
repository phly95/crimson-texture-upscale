"""Recipe for dumhit: cockpit canopy glass impact fracture / shield hit, with
conchoidal fracture shock lobes, penetrating radial fissure, specular reflection ridges,
and refractive glass stress shading.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import polar_flow, ridge_noise, calibrate_alpha, calibrate_color

def make_dumhit():
    W, H = 512, 512
    orig = fx.load("sprites/dumhit/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Base spatial envelope
    orig_up = cv2.resize(orig, (W, H), interpolation=cv2.INTER_LANCZOS4)
    env_a = np.clip(orig_up[..., 3], 0, 1)

    cx, cy = 0.48, 0.48
    r, theta, (tx, ty), (rx, ry) = polar_flow(W, H, cx=cx, cy=cy)
    x, y = fx.grid(W, H)

    # 1. Conchoidal glass fracture ridges
    cracks1 = ridge_noise(W, H, scale=24, octaves=4, seed=11, gamma=3.0)
    cracks2 = ridge_noise(W, H, scale=12, octaves=3, seed=22, gamma=3.5)

    # 2. Prominent vertical penetrating fissure (from top rim to center)
    fissure_dist = np.abs(x - 0.48 + (y - 0.30) * 0.15)
    fissure_len = fx.smoothstep(0.48, 0.20, y) * fx.smoothstep(0.12, 0.22, y)
    fissure_line = np.exp(-fissure_dist ** 2 / (2.0 * (0.008 ** 2))) * fissure_len

    # 3. Glass rim caustic / specular reflection ridge
    rim_band = fx.smoothstep(0.25, 0.38, r) * (1.0 - fx.smoothstep(0.38, 0.46, r))
    specular_rim = rim_band * (0.6 + 0.4 * np.cos(theta - np.pi/4.0)) ** 2.0

    # 4. Detail modulation
    detail = (
        0.70
        + 0.25 * cracks1
        + 0.20 * cracks2
        + 0.35 * fissure_line
        + 0.25 * specular_rim
    )

    presence = fx.smoothstep(0.015, 0.25, env_a)
    a_hd = env_a * ((1.0 - presence) + presence * detail)

    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=True, iterations=2)

    # 5. Shading: Glass refraction & specular highlight
    specular_intensity = np.clip(
        env_a * 0.6 + fissure_line * 0.5 + cracks1 * 0.25 + specular_rim * 0.35,
        0, 1
    )

    stops = [
        (0.00, (0.18, 0.20, 0.26)),   # Deep glass edge shadow
        (0.35, (0.32, 0.35, 0.44)),   # Glass body blue-grey
        (0.65, (0.62, 0.65, 0.72)),   # Refractive light blue
        (0.85, (0.85, 0.88, 0.93)),   # Specular caustic
        (1.00, (0.95, 0.96, 0.99))    # Pure white specular crest
    ]
    rgb_hd = fx.ramp(specular_intensity, stops)
    rgb_final = calibrate_color(rgb_hd, rgb_orig, a_orig, iterations=1)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/dumhit")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/dumhit/draw.png")

if __name__ == "__main__":
    make_dumhit()
