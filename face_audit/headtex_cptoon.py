"""headtex_cptoon.py TARGET [steps]: cartoon version of headtex_run for the cockpit pilots. Renders of the original head from 7 views (headtex/v_<tg>_<view>.png, from headtex_run)
are klein-redrawn as a clean cartoon, aligned + colour-locked to the render, then baked back into the UV sheet (headtex_bake) -> headtex/<tg>_toon_bake.png"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_bake as B, headtex_fin as F, headtex_run as R
T = F.T; D = R.D; PW = 60.0
STYLE = (" Redraw this render as a clean, stylised cartoon in the manner of a hand-painted 1930s pulp-adventure video game: smooth simplified painted shapes, soft cel shading, bold clean readable features, "
         "no pores, no stubble, no photographic texture. Keep every shape, colour, position and outline exactly the same; do not re-frame, re-scale, rotate or move anything, keep the plain grey background. Flat, even lighting with no cast shadows.")
WHO = {"cphead": "the head and neck of a 1930s aviator in a smooth dark brown leather flying helmet with a chinstrap and round steel-rimmed goggles with dark teal lenses, warm tan cartoon skin, calm closed mouth.",
       "cphead02": "the head and neck of a 1930s aviator in a smooth dark brown leather flying helmet with big ear flaps, a strap and round steel-rimmed goggles with dark teal lenses, warm tan cartoon skin, calm closed mouth."}
NOFACE = {"back": " This is the BACK of the head seen from behind: only the smooth leather helmet with its strap and the nape of the neck are visible; there is no face, no eyes, no mouth and NO goggles anywhere.",
          "top": " This is the TOP of the head seen from directly above: only the smooth plain leather helmet crown is visible; there is no face, no eyes, no mouth and NO goggles anywhere."}
def run(tg, steps="cb", seed=7):
    C = R.TARGETS[tg]; views = []
    for k, (y, p) in R.VIEWS.items():
        src, out, fin = D / f"v_{tg}_{k}.png", D / f"gt_{tg}_{k}.png", D / f"rt_{tg}_{k}.png"
        if "c" in steps and not fin.exists():
            T.locked_gen_wh(str(src), str(out), "This is " + WHO[tg] + STYLE + NOFACE.get(k, ""), 1024, 1024, seed)
            ref = np.array(Image.open(src).convert("RGB")); gen = np.array(Image.open(out).convert("RGB").resize((1024, 1024), Image.LANCZOS))
            w, fl = F.align(gen, ref, smooth=14.0, clip=30.0); fn = np.clip(T.colour_lock(w / 255., ref / 255., 10.0, 30.0), 0, 1)
            Image.fromarray((fn * 255 + .5).astype(np.uint8)).save(fin); print(tg, k, "lowcorr %.3f" % F.lowcorr((fn * 255).astype(np.uint8), ref), flush=True)
        if "b" in steps: views.append((y, p, np.array(Image.open(fin).convert("RGB")), np.load(D / f"v_{tg}_{k}_z.npy")))
    if "b" in steps:
        name, stem, (tw, th), _ = C["bake"][0]
        b, w = B.bake(views, model=C["model"], pack=C["pack"], only=name, TS=tw, TSH=th, pw=PW)
        a = (w > 1e-30).astype(np.float64)[..., None]; Image.fromarray(np.dstack([np.clip(b + .5, 0, 255), a[..., 0] * 255]).astype(np.uint8), "RGBA").save(D / f"{tg}_toon_bake.png"); print(tg, "covered %.2f" % (w > 0.05).mean())
if __name__ == "__main__": run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "cb")
