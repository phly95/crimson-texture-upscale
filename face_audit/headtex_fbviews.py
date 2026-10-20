"""headtex_fbviews.py VIEW SEEDS...: klein cartoon redraw of a side / back view of the fbhead1 head, rendered with the current composite (headtex/fbhead1_c_1024.png),
gently aligned to the render (no colour lock: the wool is meant to turn cream) -> headtex/fbv_<view>_s<seed>_fin.png (crop of an S-sized view at OFF)"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_render as HR, headtex_figure as FG, headtex_fin as F
T = F.T; D = FG.ROOT / "face_audit/headtex"; MODEL, PACK = 1124, "C1C/gamez.zip"; CW = 1024
STYLE = (" Redraw it as a clean, stylised cartoon in the manner of a hand-painted 1930s pulp-adventure video game, matching a cheerful kindly elderly man with smooth warm pink-beige skin: "
         "a thick soft cream-white woolly knitted hood covering the top, back and sides of his head, drawn as clean cel-shaded wool with a few soft rounded lobes, no grey, no stains; "
         "keep the head exactly the same size, position, outline and pose as the input, plain dark grey background, flat even lighting.")
VIEWS = {"sideL": (270, 1360, (168, 40), "This is a side view (profile) of the head of a 3D game character: the woolly hood, a simple pink ear on the side, the side of his smiling face at the front edge with thin round wire glasses, his neck and a brown coat collar below." + STYLE),
         "side": (90, 1360, (168, 40), "This is a side view (left profile) of the head of a 3D game character: the woolly hood, a simple pink ear on the side, the side of his smiling face at the front edge with round wire glasses, his neck and a brown tweed coat collar below." + STYLE),
         "back": (0, 1456, (216, 45), "This is the BACK of the head of a 3D game character seen from behind: only the back of the woolly hood and the back of the neck with a brown tweed coat collar are visible; there is no face, no eyes and no mouth anywhere." + STYLE)}
def ref(view, tex=None):
    yaw, S, (x0, y0), _ = VIEWS[view]; t = FG.texdict({"fbhead1": tex if tex is not None else np.array(Image.open(D / ("fbhead1_c2_1024.png" if view == "sideL" else "fbhead1_c_1024.png")).convert("RGB"))})
    im = HR.render(t, yaw, S=S, flipv=False, model=MODEL, pack=PACK, engine=True); c = im[y0:y0 + CW, x0:x0 + CW]; Image.fromarray(c).save(D / f"fbv_{view}_ref.png"); return c
def gen(view, seeds):
    r = ref(view)
    for s in seeds:
        out = D / f"fbv_{view}_s{s}.png"; T.locked_gen_wh(str(D / f"fbv_{view}_ref.png"), str(out), VIEWS[view][3], CW, CW, s)
        g = np.array(Image.open(out).convert("RGB").resize((CW, CW), Image.LANCZOS)); w, fl = F.align(g, r, smooth=20.0, clip=15.0)
        Image.fromarray(w).save(D / f"fbv_{view}_s{s}_fin.png"); print(view, s, "lowcorr raw %.3f aligned %.3f" % (F.lowcorr(g, r), F.lowcorr(w, r)), flush=True)
if __name__ == "__main__": gen(sys.argv[1], [int(x) for x in sys.argv[2:]])
