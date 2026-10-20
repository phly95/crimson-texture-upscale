"""headtex_cpsuit.py: hand-drawn pilot jacket sheets. The klein own-layout redraw (headtex_toonuv raw output, headtex/<stem>_s<seed>.png) is registered to the original with the smooth flow (layout), and only its very broad Lab tone
is nudged toward the original (large sigma, partial lightness): the drawn palette and detail are kept (the fine colour lock of headtex_fin washes them back to the photo) -> headtex/<stem>_drawn.png"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_fin as F
T = F.T; D = F.H.ROOT / "face_audit/headtex"
PICK = {"cpsuit128": 1, "cpsuit128_01": 1, "cpsuit128_02": 2}
def make(stem, seed, sl=60.0, sab=80.0, k=0.5):
    ref = np.array(Image.open(D / f"{stem}_ref.png").convert("RGB")); gen = np.array(Image.open(D / f"{stem}_s{seed}.png").convert("RGB").resize(ref.shape[1::-1], Image.LANCZOS))
    w, fl = F.align(gen, ref, smooth=20.0, clip=40.0); out = np.clip(T.colour_lock(w / 255., ref / 255., sl, sab, k_l=k), 0, 1)
    img = (out * 255 + .5).astype(np.uint8); Image.fromarray(img).save(D / f"{stem}_drawn.png"); print(stem, seed, "layout lowcorr %.3f" % F.lowcorr(img, ref)); return img
if __name__ == "__main__":
    for st, sd in PICK.items(): make(st, sd)
