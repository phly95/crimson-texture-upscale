"""headtex_bakeall.py: bake the redrawn views (r_*.png) into the cphead UV layout -> headtex/cphead_baked.png (gaps filled from the layout-locked redraw)"""
import sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_bake as B, headtex_render as HR
D = Path(__file__).resolve().parent / "headtex"
VIEWS = {"front": (180, 0), "right": (90, 0), "left": (270, 0), "back": (0, 0), "top": (0, 70), "q1": (135, 15), "q2": (225, 15)}
views = [(y, p, np.array(Image.open(D / f"r_{k}.png").convert("RGB")), np.load(D / f"v_{k}_z.npy")) for k, (y, p) in VIEWS.items()]
b, w = B.bake(views, pw=3.0)
fb = np.array(Image.open(D / "cphead_s2_fin.png").convert("RGB")).astype(np.float64)
a = np.clip(w / 0.25, 0, 1)[..., None]; a = cv2.GaussianBlur(a, (0, 0), 1.0)[..., None] if a.ndim == 3 and False else a
out = b * a + fb * (1 - a)
Image.fromarray(np.clip(out + .5, 0, 255).astype(np.uint8)).save(D / "cphead_baked.png")
print("covered %.2f" % (w > 0.05).mean())
tex = np.clip(out, 0, 255).astype(np.uint8); orig = np.array(Image.open(D.parents[1] / "pipeline/units/cphead__ba5c62c5/orig.png").convert("RGB"))
row = lambda t: np.hstack([HR.render(t, y, flipv=False, S=360) for y in (180, 135, 90)])
Image.fromarray(np.vstack([row(orig), row(tex)])).save(D / "bake_render_cmp.png")
