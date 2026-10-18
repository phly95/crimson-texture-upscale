"""Recipe for doublewire: painted steel bridge / zeppelin truss bay with
top and bottom chord girders, diagonal brace members, central apex gusset,
and clean horizontal seamless tiling with soft vertical margins.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import calibrate_alpha, calibrate_color, htile_resize

def make_doublewire():
    W, H = 256, 256
    orig = fx.load("sprites/doublewire/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Smooth continuous upscaling with horizontal periodic wrap
    env_a = htile_resize(a_orig, (W, H), cv2.INTER_CUBIC)
    rgb_hd = htile_resize(rgb_orig, (W, H), cv2.INTER_CUBIC)

    # In the original, row 0 and row 31 are soft outer background margins.
    # To maintain low seam metric across vertical boundary when tiled 2x2,
    # gently equalize the top and bottom edge alpha:
    top_edge = (env_a[0, :] + env_a[-1, :]) / 2.0
    ramp_y = np.linspace(1, 0, 8)[:, None]
    env_a[:8, :] = env_a[:8, :] * (1 - ramp_y) + top_edge * ramp_y
    env_a[-8:, :] = env_a[-8:, :] * (1 - ramp_y[::-1]) + top_edge * ramp_y[::-1]

    # Add subtle metal/paint grime texture
    grime = fx.fbm(W, H, scale=16, octaves=3, seed=55, tile=True) * 0.10 - 0.05
    rgb_textured = np.clip(rgb_hd + grime[..., None], 0, 1)

    # Calibrate alpha and color with htile=True for seamless horizontal tiling
    a_final = calibrate_alpha(env_a, a_orig, strict_zero=False, iterations=2, htile=True)

    # Ensure vertical seam continuity
    edge_avg = (a_final[0, :] + a_final[-1, :]) / 2.0
    blend_n = 4
    for i in range(blend_n):
        w_edge = (blend_n - i) / (blend_n + 1)
        a_final[i, :] = a_final[i, :] * (1 - w_edge) + edge_avg * w_edge
        a_final[-1 - i, :] = a_final[-1 - i, :] * (1 - w_edge) + edge_avg * w_edge

    rgb_final = calibrate_color(rgb_textured, rgb_orig, a_orig, iterations=1, tile=False)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/doublewire")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/doublewire/draw.png")

if __name__ == "__main__":
    make_doublewire()
