"""Recipe for rail1: industrial balcony / walkway railing bay with
top rail cap, vertical post, horizontal intermediate rail, V-brace,
and weathered bronze/steel patina drawn with clean crisp geometric bevels.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import calibrate_alpha, calibrate_color

def make_rail1():
    W, H = 128, 128
    orig = fx.load("sprites/rail1/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Draw vector geometry with skia for crisp architectural edges
    def draw_rail(c, sk):
        paint_post = sk.Paint(AntiAlias=True, Color4f=sk.Color4f(0.30, 0.23, 0.18, 1.0))
        paint_rail = sk.Paint(AntiAlias=True, Color4f=sk.Color4f(0.38, 0.30, 0.22, 1.0))
        paint_brace = sk.Paint(AntiAlias=True, Color4f=sk.Color4f(0.25, 0.18, 0.15, 1.0),
                               StrokeWidth=14.0, Style=sk.Paint.kStroke_Style)

        # 1. Left vertical post (columns 0 to ~32 in 128x128)
        c.drawRect(sk.Rect(0, 0, 32, 128), paint_post)

        # 2. Top rail cap (rows 0 to ~16)
        c.drawRect(sk.Rect(0, 0, 128, 16), paint_rail)

        # 3. Horizontal intermediate rail (rows ~60 to ~76)
        c.drawRect(sk.Rect(32, 60, 128, 76), paint_rail)

        # 4. Diagonal V-brace: from (32, 70) to (80, 128), and from (80, 128) to (128, 70)
        p = sk.Path()
        p.moveTo(28, 68)
        p.lineTo(80, 124)
        p.lineTo(132, 68)
        c.drawPath(p, paint_brace)

    vec_layer = fx.paint(W, H, draw_rail)

    # Blend with smoothly upscaled original for subtle bevel and lighting gradients
    orig_up = cv2.resize(orig, (W, H), interpolation=cv2.INTER_LANCZOS4)
    a_up = np.clip(orig_up[..., 3], 0, 1)
    rgb_up = np.clip(orig_up[..., :3], 0, 1)

    # Sharpen alpha by combining vector mask and envelope
    a_hd = np.clip(vec_layer[..., 3] * 0.4 + a_up * 0.6, 0, 1)
    # Enhance contrast of the silhouette slightly
    a_hd = fx.smoothstep(0.15, 0.75, a_hd)

    # Metal texture & specular bevels
    grime = fx.fbm(W, H, scale=12, octaves=3, seed=33) * 0.12 - 0.06
    rgb_hd = np.clip(rgb_up * 0.7 + vec_layer[..., :3] * 0.3 + grime[..., None], 0, 1)

    # Calibrate alpha and color
    a_final = calibrate_alpha(a_hd, a_orig, strict_zero=True, iterations=2)
    rgb_final = calibrate_color(rgb_hd, rgb_orig, a_orig, iterations=1)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/rail1")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/rail1/draw.png")

if __name__ == "__main__":
    make_rail1()
