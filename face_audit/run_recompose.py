import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit")
import regen as R, numpy as np
from PIL import Image, ImageDraw
f = lambda sfx: [u for u in R.last if u.endswith(sfx)][0]
OUT = R.ROOT / "face_audit/out3"; OUT.mkdir(exist_ok=True)
JOBS = [  # target, anchor
 ("29b2441a", "bd1b6b98"), ("48a14f25", "bd1b6b98"), ("8a6b3827", "bd1b6b98"),
 ("cd32a638", "3a773a4e"),
 ("150da4a0", "0110fec6"), ("1955c2f3", "0110fec6"), ("30bd314f", "0110fec6"),
 ("0539e4b2", "2449f2ca"),
 ("546bd786", "ac5acf90"),
]
sel = set(sys.argv[1:]); rows = []
for t, a in JOBS:
    if sel and t not in sel: continue
    ut, ua = f(t), f(a)
    out, msg = R.recompose(ua, ut); print(t, "<-", a, msg, flush=True)
    if out is None: continue
    Image.fromarray(out, "RGBA").save(OUT / f"{ut}.png"); rows.append((t, a, R.chosen(ut, "chosen"), out, R.chosen(ua, "chosen")))
for k in range(0, len(rows), 3):
    part = rows[k:k + 3]; sh = Image.new("RGB", (3 * 410, len(part) * 440), (30, 30, 30)); d = ImageDraw.Draw(sh)
    for i, (t, a, b, o, c) in enumerate(part):
        for j, im in enumerate((b, o, c)):
            I = Image.fromarray(im[..., :3]); w, h = I.size; s = min(400 / w, 420 / h); sh.paste(I.resize((int(w * s), int(h * s))), (j * 410, i * 440))
        d.text((3, i * 440 + 422), f"{t} before | after | anchor {a}", fill=(255, 255, 0))
    sh.save(R.ROOT / f"face_audit/recomp_sheet_{k // 3}.png")
