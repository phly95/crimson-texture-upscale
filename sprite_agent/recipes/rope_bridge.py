"""Recipe for rope_bridge: suspension footbridge section with wooden vertical stanchion post,
catenary handrail rope, inclined stay ropes, timber deck walkway, and natural hemp twist texture
rendered with clean crisp geometry.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import calibrate_alpha, calibrate_color

def make_rope_bridge():
    W, H = 256, 256
    orig = fx.load("sprites/rope_bridge/orig.png")
    a_orig = orig[..., 3]
    rgb_orig = orig[..., :3]

    # Draw vector geometry with skia
    def draw_bridge(c, sk):
        paint_post = sk.Paint(AntiAlias=True, Color4f=sk.Color4f(0.35, 0.30, 0.22, 1.0),
                              StrokeWidth=14.0, Style=sk.Paint.kStroke_Style, StrokeCap=sk.Paint.kRound_Cap)
        paint_rope = sk.Paint(AntiAlias=True, Color4f=sk.Color4f(0.52, 0.46, 0.35, 1.0),
                              StrokeWidth=10.0, Style=sk.Paint.kStroke_Style, StrokeCap=sk.Paint.kRound_Cap)
        paint_deck = sk.Paint(AntiAlias=True, Color4f=sk.Color4f(0.40, 0.35, 0.25, 1.0),
                              StrokeWidth=12.0, Style=sk.Paint.kStroke_Style)

        # Central post
        c.drawLine(128, 0, 128, 256, paint_post)

        # Deck
        c.drawLine(0, 250, 256, 250, paint_deck)

        # Catenary top handrail
        p_hand = sk.Path()
        p_hand.moveTo(0, 40)
        p_hand.quadTo(64, 30, 128, 6)
        p_hand.quadTo(192, 30, 256, 40)
        c.drawPath(p_hand, paint_rope)

        # Inclined stay ropes
        p_stay = sk.Path()
        p_stay.moveTo(128, 60)
        p_stay.quadTo(80, 150, 24, 250)
        c.drawPath(p_stay, paint_rope)

        p_stay2 = sk.Path()
        p_stay2.moveTo(128, 60)
        p_stay2.quadTo(176, 150, 232, 250)
        c.drawPath(p_stay2, paint_rope)

    vec_layer = fx.paint(W, H, draw_bridge)

    # Smooth continuous upscaling using Lanczos
    orig_up = cv2.resize(orig, (W, H), interpolation=cv2.INTER_LANCZOS4)
    a_up = np.clip(orig_up[..., 3], 0, 1)
    rgb_up = np.clip(orig_up[..., :3], 0, 1)

    # Combine vector shape with envelope
    a_comb = np.clip(vec_layer[..., 3] * 0.4 + a_up * 0.6, 0, 1)
    a_comb = fx.smoothstep(0.12, 0.78, a_comb)

    # Add subtle wood grain / rope fiber texture
    fiber = fx.fbm(W, H, scale=8, octaves=3, seed=77) * 0.10 - 0.05
    rgb_textured = np.clip(rgb_up * 0.7 + vec_layer[..., :3] * 0.3 + fiber[..., None], 0, 1)

    # Calibrate alpha and color
    a_final = calibrate_alpha(a_comb, a_orig, strict_zero=True, iterations=2)
    rgb_final = calibrate_color(rgb_textured, rgb_orig, a_orig, iterations=1)

    layer = np.dstack([np.clip(rgb_final, 0, 1), np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/rope_bridge")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/rope_bridge/draw.png")

if __name__ == "__main__":
    make_rope_bridge()
