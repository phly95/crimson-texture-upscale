"""headtex_fbface.py STEP [seeds]: fbhead1 face via a FRONT portrait: render the head front-on with the original texture (collapsed right-half UVs borrowed from the mirror side),
klein-redraw it as a cartoon portrait, align + colour-lock to the render -> headtex/fbp_s<seed>_fin.png (crop of an S=1600 view, offset OFF)"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_render as HR, headtex_figure as FG, headtex_fin as F
T = F.T; D = FG.ROOT / "face_audit/headtex"; MODEL, PACK = 1124, "C1C/gamez.zip"; S = 1456; OFF = (216, 45); CW = 1024   # crop x0, y0, size
PROMPT = ("This is a front view of the head of a 3D game character. Redraw it as a clean, stylised cartoon portrait in the manner of a hand-painted 1930s pulp-adventure video game: "
          "a cheerful, kindly elderly man looking straight ahead, perfectly symmetric face, two friendly dark brown eyes with white highlights, soft white eyebrows, a rounded nose, a warm closed-mouth smile showing a little clean white teeth, "
          "smooth warm pink-beige skin with soft cel shading, no wrinkles, no stubble, no pores; thin round dark wire glasses over both eyes; a thick white woolly knitted cap covering his head, ears and the sides of his face, soft clean off-white wool; "
          "a pink neck below. Keep the head exactly the same size, position, outline and pose as the input, plain dark grey background, flat even front lighting.")
def ref():
    t = FG.texdict(); im = HR.render(t, 180, S=S, flipv=False, model=MODEL, pack=PACK, fixuv=True, engine=True)
    x0, y0 = OFF; c = im[y0:y0 + CW, x0:x0 + CW]; Image.fromarray(c).save(D / "fbp_ref.png"); return c
def gen(seeds):
    r = ref()
    for s in seeds:
        out = D / f"fbp_s{s}.png"; T.locked_gen_wh(str(D / "fbp_ref.png"), str(out), PROMPT, CW, CW, s)
        g = np.array(Image.open(out).convert("RGB").resize((CW, CW), Image.LANCZOS)); w, fl = F.align(g, r, smooth=14.0, clip=30.0)
        fn = np.clip(T.colour_lock(w / 255., r / 255., 10.0, 30.0), 0, 1); Image.fromarray((fn * 255 + .5).astype(np.uint8)).save(D / f"fbp_s{s}_fin.png")
        print(s, "lowcorr raw %.3f fin %.3f" % (F.lowcorr(g, r), F.lowcorr((fn * 255).astype(np.uint8), r)), flush=True)
if __name__ == "__main__":
    if sys.argv[1] == "ref": ref()
    else: gen([int(x) for x in sys.argv[2:]])
