"""Recipe for ladder_01: industrial vertical access ladder section with
cylindrical steel side stiles, tubular horizontal rungs, cylindrical bevel shading,
and seamless vertical tiling rendered with crisp architectural geometry.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import calibrate_alpha, calibrate_color, vtile_resize

def make_ladder_01():
    W, H = 256, 256
    orig = fx.load("sprites/ladder_01/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Draw vector geometry with skia
    def draw_ladder(c, sk):
        paint_stile = sk.Paint(AntiAlias=True, Color4f=sk.Color4f(0.50, 0.53, 0.52, 1.0))
        paint_rung = sk.Paint(AntiAlias=True, Color4f=sk.Color4f(0.42, 0.45, 0.44, 1.0))

        # Left stile (columns ~0 to ~32)
        c.drawRect(sk.Rect(0, 0, 32, 256), paint_stile)
        # Right stile (columns ~224 to ~256)
        c.drawRect(sk.Rect(224, 0, 256, 256), paint_stile)

        # Middle horizontal rung (rows ~96 to ~128)
        c.drawRect(sk.Rect(32, 96, 224, 128), paint_rung)
        # Bottom partial rung (rows ~224 to ~256)
        c.drawRect(sk.Rect(32, 224, 224, 256), paint_rung)

    vec_layer = fx.paint(W, H, draw_ladder)

    # Smooth continuous upscaling with vertical periodic wrap
    env_a = vtile_resize(a_orig, (W, H), cv2.INTER_CUBIC)
    rgb_hd = vtile_resize(rgb_orig, (W, H), cv2.INTER_CUBIC)

    # Combine vector shape with envelope for crisp cylindrical bevels
    a_comb = np.clip(vec_layer[..., 3] * 0.4 + env_a * 0.6, 0, 1)
    a_comb = fx.smoothstep(0.12, 0.78, a_comb)

    # Add subtle brushed steel / galvanized texture
    grime = fx.fbm(W, H, scale=12, octaves=3, seed=66, tile=True) * 0.08 - 0.04
    rgb_textured = np.clip(rgb_hd * 0.7 + vec_layer[..., :3] * 0.3 + grime[..., None], 0, 1)

    # Calibrate alpha and color with vtile=True for seamless vertical tiling
    a_final = calibrate_alpha(a_comb, a_orig, strict_zero=False, iterations=2, vtile=True)
    rgb_final = calibrate_color(rgb_textured, rgb_orig, a_orig, iterations=1, tile=False)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/ladder_01")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/ladder_01/draw.png")

if __name__ == "__main__":
    make_ladder_01()
