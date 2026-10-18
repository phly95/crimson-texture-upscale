"""patch_text.py BASE4X SRC LAYER OUT X0,Y0,X1,Y1 [...]   (boxes in SOURCE pixels): targeted lettering repair on a finished 4x picture.
Inside each box the redraw's lettering is removed (every pixel near the original's text, per the text layer from textlayer.split, is inpainted from the surrounding paper of the 4x picture)
and the ORIGINAL lettering is laid back on at 4x (alpha upscaled and edge-sharpened, colour unmixed). Letters are therefore exactly the original's, slightly softer than a typeset font;
the picture outside the boxes is untouched. Box edges are feathered."""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import textlayer as tl


def patch(base, layer, boxes, scale=4, grow=7, sharp=2.4, feather=10):
    W, H = base.size
    arr = np.asarray(base.convert("RGB")).copy()
    ly = layer.convert("RGBA")
    a4 = np.asarray(ly.getchannel("A").resize((W, H), Image.LANCZOS), np.float32) / 255
    c4 = np.asarray(ly.convert("RGB").resize((W, H), Image.BICUBIC), np.float32)
    a_s = np.clip((a4 - 0.5) * sharp + 0.5, 0, 1)
    for (x0, y0, x1, y1) in boxes:
        X0, Y0, X1, Y1 = [int(v * scale) for v in (x0, y0, x1, y1)]
        sub = arr[Y0:Y1, X0:X1]
        m = (a4[Y0:Y1, X0:X1] > 0.04).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
        clean = cv2.inpaint(cv2.cvtColor(sub, cv2.COLOR_RGB2BGR), m * 255, 5, cv2.INPAINT_TELEA)
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB).astype(np.float32)
        a = a_s[Y0:Y1, X0:X1, None]
        new = clean * (1 - a) + c4[Y0:Y1, X0:X1] * a
        fm = np.zeros(sub.shape[:2], np.float32); fm[feather:-feather or None, feather:-feather or None] = 1
        fm = cv2.GaussianBlur(fm, (0, 0), feather / 2.5)[..., None]
        arr[Y0:Y1, X0:X1] = (sub * (1 - fm) + new * fm).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


if __name__ == "__main__":
    base, src, layer, out = sys.argv[1:5]
    boxes = [tuple(float(v) for v in b.split(",")) for b in sys.argv[5:]]
    patch(Image.open(base), Image.open(layer), boxes).save(out); print("patched", len(boxes), "boxes ->", out)
