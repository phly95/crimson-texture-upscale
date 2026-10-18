"""Recipe for eiffel1: monumental wrought-iron / steel truss lattice panel with
perimeter frame, vertical and horizontal crossbeams, diagonal X-bracing,
riveted hub node, sage/olive green painted patina, and seamless 2D tiling.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import calibrate_alpha, calibrate_color, periodic_resize

def make_eiffel1():
    W, H = 512, 512
    orig = fx.load("sprites/eiffel1/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Smooth continuous upscaling with 2D periodic wrapping
    env_a = periodic_resize(a_orig, (W, H), cv2.INTER_CUBIC)
    rgb_hd = periodic_resize(rgb_orig, (W, H), cv2.INTER_CUBIC)

    # In the original, the frame edges wrap seamlessly.
    # Equalize the opposing perimeter edges so edge difference matches zero seam:
    edge_y = (env_a[0, :] + env_a[-1, :]) / 2.0
    edge_x = (env_a[:, 0] + env_a[:, -1]) / 2.0

    blend_n = 4
    for i in range(blend_n):
        w = (blend_n - i) / (blend_n + 1)
        env_a[i, :] = env_a[i, :] * (1 - w) + edge_y * w
        env_a[-1 - i, :] = env_a[-1 - i, :] * (1 - w) + edge_y * w
        env_a[:, i] = env_a[:, i] * (1 - w) + edge_x * w
        env_a[:, -1 - i] = env_a[:, -1 - i] * (1 - w) + edge_x * w

    # Same for rgb
    edge_rgb_y = (rgb_hd[0, :, :] + rgb_hd[-1, :, :]) / 2.0
    edge_rgb_x = (rgb_hd[:, 0, :] + rgb_hd[:, -1, :]) / 2.0
    for i in range(blend_n):
        w = (blend_n - i) / (blend_n + 1)
        rgb_hd[i, :, :] = rgb_hd[i, :, :] * (1 - w) + edge_rgb_y * w
        rgb_hd[-1 - i, :, :] = rgb_hd[-1 - i, :, :] * (1 - w) + edge_rgb_y * w
        rgb_hd[:, i, :] = rgb_hd[:, i, :] * (1 - w) + edge_rgb_x * w
        rgb_hd[:, -1 - i, :] = rgb_hd[:, -1 - i, :] * (1 - w) + edge_rgb_x * w

    # Add subtle metal/paint grime texture
    grime = fx.fbm(W, H, scale=16, octaves=4, seed=88, tile=True) * 0.08 - 0.04
    rgb_textured = np.clip(rgb_hd + grime[..., None], 0, 1)

    # Calibrate alpha and color with tile=True for seamless 2D tiling
    a_final = calibrate_alpha(env_a, a_orig, strict_zero=False, iterations=2, tile=True)

    # Enforce seamless boundaries on final alpha
    edge_y_f = (a_final[0, :] + a_final[-1, :]) / 2.0
    edge_x_f = (a_final[:, 0] + a_final[:, -1]) / 2.0
    for i in range(blend_n):
        w = (blend_n - i) / (blend_n + 1)
        a_final[i, :] = a_final[i, :] * (1 - w) + edge_y_f * w
        a_final[-1 - i, :] = a_final[-1 - i, :] * (1 - w) + edge_y_f * w
        a_final[:, i] = a_final[:, i] * (1 - w) + edge_x_f * w
        a_final[:, -1 - i] = a_final[:, -1 - i] * (1 - w) + edge_x_f * w

    rgb_final = calibrate_color(rgb_textured, rgb_orig, a_orig, iterations=1, tile=True)

    # Enforce seamless boundaries on final rgb
    edge_rgb_yf = (rgb_final[0, :, :] + rgb_final[-1, :, :]) / 2.0
    edge_rgb_xf = (rgb_final[:, 0, :] + rgb_final[:, -1, :]) / 2.0
    for i in range(blend_n):
        w = (blend_n - i) / (blend_n + 1)
        rgb_final[i, :, :] = rgb_final[i, :, :] * (1 - w) + edge_rgb_yf * w
        rgb_final[-1 - i, :, :] = rgb_final[-1 - i, :, :] * (1 - w) + edge_rgb_yf * w
        rgb_final[:, i, :] = rgb_final[:, i, :] * (1 - w) + edge_rgb_xf * w
        rgb_final[:, -1 - i, :] = rgb_final[:, -1 - i, :] * (1 - w) + edge_rgb_xf * w

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/eiffel1")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/eiffel1/draw.png")

if __name__ == "__main__":
    make_eiffel1()
