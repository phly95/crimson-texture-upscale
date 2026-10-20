"""headtex_toon.py TARGET [VIEW SEED]: cartoon-style variant of the mesh-view redraw. Restyles each reference render of the head mesh (original texture) as a clean hand-painted cartoon with klein, aligned + colour-locked to the render
-> headtex/t_<tg>_<view>[_s<seed>].png ; then headtex_toon bake via TOON=1 in headtex_run-like call (see bake())"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_run as R, headtex_fin as F
T, D = R.T, R.D
STYLE = ("Redraw this render as a clean, stylised cartoon character in the manner of a hand-painted 1930s pulp-adventure video game: smooth simplified shapes, soft painterly cel shading with gentle highlights, "
         "clean readable features, no pores, no wrinkles, no photographic texture. Keep every shape, colour, position and outline exactly the same; do not re-frame, re-scale, rotate or move anything, keep the plain grey background. ")
WHO = {"cphead": "A pilot's head and neck in a dark brown leather flying helmet with a chinstrap and round goggles.", "cphead02": "A pilot's head and neck in a dark brown leather flying helmet with big ear flaps, a chinstrap and round goggles.",
       "woman": "A smiling young woman's head and neck with dark brown skin, wearing a snug dark leather skull cap; smooth hairless skin, bright white teeth.", "fbhead1": "A cheerful elderly man's head and neck with round glasses and a white woolly cap."}
TAIL = {"front": " The face is perfectly symmetrical with a straight nose and a calm, friendly, neutral mouth; both goggle lenses are the same size and sit side by side over the eyes.", **R.NOFACE}
def gen(tg, k, seed=7, tail_extra=""):
    src = D / f"v_{tg}_{k}.png"; out = D / f"gt_{tg}_{k}_s{seed}.png"; fin = D / f"t_{tg}_{k}_s{seed}.png"
    if not fin.exists():
        T.locked_gen_wh(str(src), str(out), STYLE + WHO[tg] + TAIL.get(k, "") + tail_extra, 1024, 1024, seed)
        ref = np.array(Image.open(src).convert("RGB")); g = np.array(Image.open(out).convert("RGB").resize((1024, 1024), Image.LANCZOS)); w, fl = F.align(g, ref, smooth=14.0, clip=30.0)
        Image.fromarray((np.clip(T.colour_lock(w / 255., ref / 255., 10.0, 30.0), 0, 1) * 255 + .5).astype(np.uint8)).save(fin)
    print(tg, k, seed, "lowcorr %.3f" % F.lowcorr(np.array(Image.open(fin).convert("RGB")), np.array(Image.open(src).convert("RGB"))), flush=True); return fin
if __name__ == "__main__":
    tg = sys.argv[1]
    if len(sys.argv) > 2: [gen(tg, sys.argv[2], s) for s in map(int, sys.argv[3:])]
    else: [gen(tg, k) for k in R.VIEWS]
