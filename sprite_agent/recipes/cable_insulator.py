"""Recipe for cable_insulator: electrical high-voltage ceramic disc insulator
stack with tapered shed geometry, central mounting spindle, and clean anti-aliasing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx
from recipes.common import calibrate_alpha, vtile_resize

def make_cable_insulator():
    W, H = 256, 256
    orig = fx.load("sprites/cable_insulator/orig.png")
    a_orig = orig[..., 3]

    # Draw vector geometry with skia for crisp ceramic sheds
    def draw_insulator(c, sk):
        paint_black = sk.Paint(AntiAlias=True, Color4f=sk.Color4f(0, 0, 0, 1.0))

        # Central spindle rod: from top to bottom, width 12 px (centered at 128)
        c.drawRect(sk.Rect(122, 0, 134, 256), paint_black)

        # Bottom knob/terminal cap (row 31 in orig, y ~ 240..252)
        c.drawOval(sk.Rect(120, 240, 136, 252), paint_black)

        # 6 tapered shed discs, from top to bottom
        # Heights and widths measured from orig.png:
        # 1. y ~ 16..28, width ~ 140 (x 58..198)
        # 2. y ~ 48..64, width ~ 124 (x 66..190)
        # 3. y ~ 84..100, width ~ 110 (x 73..183)
        # 4. y ~ 124..140, width ~ 96 (x 80..176)
        # 5. y ~ 164..180, width ~ 80 (x 88..168)
        # 6. y ~ 204..220, width ~ 64 (x 96..160)

        sheds = [
            (16, 26, 136),
            (48, 62, 122),
            (84, 98, 108),
            (124, 138, 92),
            (164, 178, 76),
            (204, 218, 60),
        ]

        for y0, y1, width in sheds:
            # Trapeze / umbrella shed
            p = sk.Path()
            p.moveTo(128 - width / 2, y1)
            p.lineTo(128 - width / 2 + 4, y0 + 3)
            p.quadTo(128, y0 - 2, 128 + width / 2 - 4, y0 + 3)
            p.lineTo(128 + width / 2, y1)
            p.close()
            c.drawPath(p, paint_black)

    vec_layer = fx.paint(W, H, draw_insulator)

    # Resample original envelope
    orig_up = cv2.resize(a_orig, (W, H), interpolation=cv2.INTER_LANCZOS4)
    orig_up = np.clip(orig_up, 0, 1)

    # Combine vector precision with original envelope
    a_comb = np.clip(vec_layer[..., 3] * 0.5 + orig_up * 0.5, 0, 1)
    a_comb = fx.smoothstep(0.12, 0.75, a_comb)

    # Match central spindle width across top and bottom for seamless vertical continuity
    spindle_mask = (a_comb[0, :] > 0.05) | (a_comb[-1, :] > 0.05)
    joint_edge = (a_comb[0, :] + a_comb[-1, :]) / 2.0

    blend_rows = 6
    for i in range(blend_rows):
        w = (blend_rows - i) / (blend_rows + 1)
        a_comb[i, :] = np.where(spindle_mask, a_comb[i, :] * (1 - w) + joint_edge * w, a_comb[i, :])
        a_comb[-1 - i, :] = np.where(spindle_mask, a_comb[-1 - i, :] * (1 - w) + joint_edge * w, a_comb[-1 - i, :])

    # Calibrate alpha
    a_final = calibrate_alpha(a_comb, a_orig, strict_zero=False, iterations=2, vtile=True)

    # Final boundary matching
    joint_final = (a_final[0, :] + a_final[-1, :]) / 2.0
    for i in range(blend_rows):
        w = (blend_rows - i) / (blend_rows + 1)
        a_final[i, :] = np.where(spindle_mask, a_final[i, :] * (1 - w) + joint_final * w, a_final[i, :])
        a_final[-1 - i, :] = np.where(spindle_mask, a_final[-1 - i, :] * (1 - w) + joint_final * w, a_final[-1 - i, :])

    # Pure black color for insulator silhouette
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    layer = np.dstack([rgb, np.clip(a_final, 0, 1)])
    layer[..., :3] *= (layer[..., 3:] > 0)

    out_dir = Path("sprites/cable_insulator")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/cable_insulator/draw.png")

if __name__ == "__main__":
    make_cable_insulator()
