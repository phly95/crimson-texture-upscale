"""headtex_views.py [names...]: klein redraw of the rendered head views (v_<name>.png -> r_<name>.png), aligned + colour-locked to the render"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_fin as F
T = F.T; D = F.H.ROOT / "face_audit/headtex"
PROMPT = ("Restore this render as a crisp, sharp, high-resolution, photorealistic version of itself: the head and neck of a 1930s aviator in a stitched dark brown leather flying helmet with chinstrap and round goggles, "
          "realistic skin and leather. Keep every shape, colour, position and outline exactly the same; do not re-frame, re-scale, rotate or move anything, keep the plain grey background, add only fine realistic surface detail "
          "and remove blur and pixelation. Flat, even lighting with no cast shadows.")
TAIL = {"back": " This is the BACK of the head seen from behind: only the leather helmet back, its strap and the nape of the neck are visible; there is no face, no eyes, no goggles and no mouth anywhere.",
        "top": " This is the TOP of the head seen from directly above: only the stitched leather helmet crown and the tip of a nose-less forehead edge are visible; there is no face, no eyes, no goggles and no mouth anywhere."}
for name in (sys.argv[1:] or "front right left back top q1 q2".split()):
    src = D / f"v_{name}.png"; out = D / f"g_{name}.png"
    T.locked_gen_wh(str(src), str(out), PROMPT + TAIL.get(name, ""), 1024, 1024, 7)
    ref = np.array(Image.open(src).convert("RGB")); gen = np.array(Image.open(out).convert("RGB").resize((1024, 1024), Image.LANCZOS))
    w, fl = F.align(gen, ref, smooth=14.0, clip=30.0)
    fin = np.clip(T.colour_lock(w / 255., ref / 255., 10.0, 30.0), 0, 1); Image.fromarray((fin * 255 + .5).astype(np.uint8)).save(D / f"r_{name}.png")
    print(name, "lowcorr raw %.3f final %.3f" % (F.lowcorr(gen, ref), F.lowcorr((fin * 255).astype(np.uint8), ref)), flush=True)
