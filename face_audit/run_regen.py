import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit")
import regen as R, numpy as np
from PIL import Image, ImageDraw
ROOT = R.ROOT; OUT = ROOT / "face_audit/out2"; OUT.mkdir(exist_ok=True)
f = lambda sfx: [u for u in R.last if u.endswith(sfx)][0]
# (target suffix, anchor suffix, centre, kwargs)
CASES = [
 ("150da4a0", "0110fec6", (0.56, 0.20), {"anchor_base": "chosen", "k": (1.5, 1.3)}),
 ("1955c2f3", "0110fec6", (0.58, 0.23), {"anchor_base": "chosen", "k": (1.5, 1.4)}),
 ("30bd314f", "0110fec6", (0.62, 0.23), {"anchor_base": "chosen", "k": (1.5, 1.4)}),
 ("cd32a638", "3a773a4e", (0.74, 0.54), {"base": "chosen", "anchor_base": "chosen", "keep_thin": True}),
 ("3f15473c", "76e22fed", (0.62, 0.17), {}),
 ("0d7277e6", "76e22fed", (0.55, 0.20), {}),
 ("1b92aa7f", "76e22fed", (0.60, 0.20), {}),
 ("6136e194", "f190a5e9", (0.20, 0.23), {}),
 ("21a6ef35", "d82b3acc", (0.56, 0.25), {}),
 ("05ea6514", "2c7cdc89", (0.22, 0.11), {"k": (1.2, 1.2)}),
 ("0539e4b2", "2449f2ca", (0.76, 0.44), {}),
 ("07f5097a", "2f32c264", (0.20, 0.40), {}),
 ("29b2441a", "bd1b6b98", (0.14, 0.24), {"anchor_base": "chosen"}),
 ("48a14f25", "bd1b6b98", (0.25, 0.24), {"anchor_base": "chosen"}),
]
sel = set(sys.argv[1:])
rows = []
for t, a, c, kw in CASES:
    if sel and t not in sel: continue
    ut, ua = f(t), f(a)
    try:
        out, msg = R.regen(ua, ut, c, **kw)
    except Exception as e:
        out, msg = None, f"error {e}"
    print(t, "<-", a, msg, flush=True)
    if out is None: continue
    Image.fromarray(out, "RGBA").save(OUT / f"{ut}.png")
    before = R.chosen(ut, kw.get("base", "latest")); anc = R.chosen(ua, kw.get("anchor_base", "latest")); og = R.og(ut)
    H, W = before.shape[:2]; r = int(0.2 * W); cx, cy = int(c[0] * W), int(c[1] * H)
    def crop(im, sc=1.0):
        h, w = im.shape[:2]; X, Y, rr = int(c[0] * w), int(c[1] * h), int(0.2 * w)
        return Image.fromarray(im[..., :3]).crop((max(0, X - rr), max(0, Y - rr), X + rr, Y + rr)).resize((300, 300), Image.LANCZOS)
    rows.append((t, crop(og), crop(before), crop(out), crop(anc)))
for k in range(0, len(rows), 4):
    part = rows[k:k + 4]
    sh = Image.new("RGB", (4 * 310, len(part) * 330), (30, 30, 30)); d = ImageDraw.Draw(sh)
    for i, (t, *ims) in enumerate(part):
        for j, im in enumerate(ims): sh.paste(im, (j * 310, i * 330))
        d.text((3, i * 330 + 305), t + "   OG | before | after | anchor", fill=(255, 255, 0))
    sh.save(ROOT / f"face_audit/regen_sheet_{k // 4}.png")
