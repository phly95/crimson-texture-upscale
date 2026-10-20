"""headtex_parts.py PART SEED... : redraw one part of the front view on its own, from a large crop of the reference (goggles | lower), mirrored to exact left-right symmetry -> headtex/part_<part>_s<seed>.png (crop coords of PARTS[part])"""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_fin as F, headtex_feat as FT
T = F.T; D = F.H.ROOT / "face_audit/headtex"
BASE = ("Keep every shape, position and size exactly the same as the input; do not add any new objects, do not add a face or text, keep the same framing. Crisp, sharp, photorealistic detail, flat even lighting, no cast shadows. ")
PARTS = {
    "gogv": ((344, 335, 680, 495), (1008, 480), "A close-up of a pair of 1930s round aviator goggles with thick black leather padding and polished steel rims, plain dark tinted glass lenses with a soft sky reflection and nothing visible behind the glass, seen straight from the front, perfectly symmetrical, on pale skin. Smooth clean rims. " + BASE),   # name: (x0, y0, x1, y1 in the 1024 front render [centred on x=512], output size, prompt)
    "goggles": ((340, 330, 684, 500), (1024, 496), "A close-up of a pair of 1930s round aviator goggles with thick black leather padding and polished steel rims, plain dark tinted glass lenses with a soft sky reflection and nothing visible behind the glass, seen straight from the front, perfectly symmetrical, with the pale skin of the brow above and the cheeks below. " + BASE),
    "lower": ((400, 430, 624, 640), (1008, 944), "A close-up of the lower half of a man's face seen straight from the front: a straight, symmetrical nose, a wide relaxed softly closed mouth with level corners, a clean-shaven chin, natural skin, a calm neutral expression, not frowning; the dark brown leather chinstrap below the chin. " + BASE),
}
def sym(img):                                     # exact left-right symmetry about the crop centre (the mesh is symmetric: crops are centred on x=512)
    h = img.shape[1] // 2; L = img[:, :h]; return np.concatenate([L, L[:, ::-1]], 1) if img.shape[1] % 2 == 0 else img
def run(part, seeds):
    (x0, y0, x1, y1), (W, H), prompt = PARTS[part]
    rp = D / f"part_{part}_ref.png"
    if part != "gogv":                                  # gogv: the clean vector drawing of headtex_vecgog.py is the reference
        ref = np.array(Image.open(D / "v_cphead_front.png").convert("RGB"))[y0:y1, x0:x1]; Image.fromarray(cv2.resize(ref, (W, H), interpolation=cv2.INTER_CUBIC)).save(rp)
    for s in seeds:
        out = D / f"part_{part}_g{s}.png"; T.locked_gen_wh(str(rp), str(out), prompt, W, H, s)
        g = np.array(Image.open(out).convert("RGB").resize((W, H), Image.LANCZOS)); r = np.array(Image.open(rp).convert("RGB"))
        w, fl = F.align(g, r, smooth=12.0, clip=40.0); fn = (np.clip(T.colour_lock(w / 255., r / 255., 14.0, 40.0), 0, 1) * 255 + .5).astype(np.uint8)
        Image.fromarray(sym(fn)).save(D / f"part_{part}_s{s}.png"); print(part, s, "lowcorr %.3f" % F.lowcorr(sym(fn), r), flush=True)
if __name__ == "__main__": run(sys.argv[1], [int(x) for x in sys.argv[2:]])
