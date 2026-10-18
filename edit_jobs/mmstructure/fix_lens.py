"""fix_lens.py: MAINMENUBUTTONSSTRUCTURE. Keep the retry:1 metal; the bottom-ring lens takes the original's translucent alpha and colour (the redraw made it a solid pale disc)."""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
U = Path("/home/deck/workspace/csvm-hd/pipeline/units/gfx_MAINMENUBUTTONSSTRUCTURE__5cdb823d")
OUT = Path(__file__).resolve().parent / "output.png"
r = Image.open(U / "retry/klein_s1_rgba.png").convert("RGBA"); W, H = r.size
a = np.asarray(r).copy()
o = Image.open(U / "orig.png").convert("RGBA").resize((W, H), Image.BICUBIC); o = np.asarray(o)
hsv = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2HSV)
box = np.zeros((H, W), bool); box[int(.72 * H):int(.93 * H), int(.40 * W):int(.62 * W)] = True
cand = box & (a[..., 3] > 200) & (hsv[..., 2] > 150) & (hsv[..., 1] < 60)
cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(cand)
k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); M = (lab == k).astype(np.uint8)
M = cv2.dilate(M, np.ones((3, 3), np.uint8), iterations=3) > 0
print("lens px", int(M.sum()), "bbox", np.nonzero(M)[1].min(), np.nonzero(M)[1].max(), np.nonzero(M)[0].min(), np.nonzero(M)[0].max())
# soft blend at the mask border so the metal rim is untouched
w = cv2.GaussianBlur(M.astype(np.float32), (0, 0), 2)[..., None] * M[..., None]
rgb = a[..., :3] * (1 - w) + o[..., :3] * w
al = a[..., 3:] * (1 - w) + o[..., 3:] * w
Image.fromarray(np.dstack([rgb, al]).astype(np.uint8), "RGBA").save(OUT)
