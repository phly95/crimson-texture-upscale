"""zach_emb.py: seeded klein redraws of the 'Fortune Hunters' emblem on the ZACHARYANDPLANE anchor (3ad54545); contact sheet + OCR score per seed."""
import sys; sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit"); sys.path.insert(0, "/home/deck/workspace/csvm-hd/tools")
import regen as R, numpy as np, cv2, re
from PIL import Image
import tile_upscale as tu, textlayer as tl
ua = [u for u in R.last if u.endswith("3ad54545")][0]
an = Image.fromarray(R.chosen(ua, "latest")[..., :3])
BOX = (0, 40, 540, 580)
crop = an.crop(BOX); ref = crop.resize((1024, 1024), Image.LANCZOS); ref.save("face_audit/zach/ref.png")
P = ("Redraw this painted nose-art emblem on the side of an aircraft as crisp, clean artwork, keeping exactly the same design and layout: a skull wearing a pirate tricorn hat with two crossed swords behind it, "
     "inside a dark oval badge with a pale outline, and the hand-lettered white brush-script title across it. The title reads exactly 'Fortune' on the first line and 'Hunters' on the second line, "
     "spelled F-o-r-t-u-n-e and H-u-n-t-e-r-s, white flowing script lettering with clean letterforms, nothing else written. "
     "Keep the metal fuselage, rivets, sky and the hand at the right exactly as in the reference. Do not add any other lettering or symbols.")
seeds = [int(s) for s in sys.argv[1:]] or list(range(300, 312))
for s in seeds:
    out = f"face_audit/zach/emb_s{s}.png"
    tu.locked_gen("face_audit/zach/ref.png", out, P, 1024, s)
    im = Image.open(out).convert("RGB")
    w = [x[0] for x in tl.ocr_words(im, 40)]
    print("seed", s, "ocr", w, flush=True)
