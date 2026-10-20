"""zach_flat.py SEEDS: can klein wrap a flat emblem onto the fuselage by itself? reference = anchor window with the flat 48gypsy pasted (small, upright) over the old badge"""
import sys, numpy as np, cv2
from PIL import Image
sys.path.insert(0, "/home/deck/workspace/csvm-hd/tools")
import os
ROT = float(os.environ.get("ROT", "0")); TAG = os.environ.get("TAG", "flat")
import tile_upscale as tu
Z = "/home/deck/workspace/csvm-hd/face_audit/zach/"
ref = np.asarray(Image.open(Z + "ref.png").convert("RGB"), np.float32)
em = np.asarray(Image.open(Z + "gypsy_hd.png").convert("RGBA"), np.float32) / 255
ys, xs = np.where(em[..., 3] > 0.05); em = em[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
k = 520 / max(em.shape[:2]); em = cv2.resize(em, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
em[..., :3] = em[..., :3].mean(-1, keepdims=True)
if ROT:
    hh, ww = em.shape[:2]; D = int(np.hypot(hh, ww)) + 4; pad = np.zeros((D, D, 4), np.float32); pad[(D - hh) // 2:(D - hh) // 2 + hh, (D - ww) // 2:(D - ww) // 2 + ww] = em
    em = cv2.warpAffine(pad, cv2.getRotationMatrix2D((D / 2, D / 2), -ROT, 1.0), (D, D), flags=cv2.INTER_LANCZOS4)
h, w = em.shape[:2]; cx, cy = 310, 440; x0, y0 = cx - w // 2, cy - h // 2
out = ref.copy(); a = em[..., 3:]
ya, yb, xa, xb = max(0, y0), min(1024, y0 + h), max(0, x0), min(1024, x0 + w)
sub = em[ya - y0:yb - y0, xa - x0:xb - x0]; aa = sub[..., 3:]
out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - aa) + sub[..., :3] * 255 * aa
Image.fromarray(out.astype(np.uint8)).save(Z + f"{TAG}_in.png")
P = ("In this black and white photo of an aircraft nose, a flat round emblem (a crystal ball with a woman holding tarot cards and the script 'Fortune Hunters') has been laid over the old badge. "
     "Paint that emblem as a decal applied to the curved fuselage: it must wrap around the curved metal like it is printed on a cylinder, tilted so that its lettering runs steeply down toward the lower right along the fuselage, with strong perspective foreshortening, "
     "so it becomes a narrow slanted oval teardrop that follows the surface and curves away toward the upper right where the fuselage turns. Keep the emblem's artwork and lettering the same, just warped onto the surface "
     "with matching light and film grain. Remove the old skull badge completely. Keep the sky, the gun barrel, the hand and the sleeve exactly as they are.")
for s in [int(a) for a in sys.argv[1:]]:
    tu.locked_gen(Z + f"{TAG}_in.png", Z + f"{TAG}_s{s}.png", P, 1024, s); print("seed", s, flush=True)
