"""alpha_tol.py STEM CAND.png [SHEET.png]: alpha metrics of a candidate against the unit's original, with the practical tolerance (see tol.py), and a contact sheet
(original | candidate on checker and dark, then both alphas and their difference). Exit 0 = within tolerance.   venv/bin/python flux_fix/alpha_tol.py cloud1 cand.png"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tol

ROOT = Path(__file__).resolve().parents[1]


def unit_of(stem):
    cat = json.load(open(ROOT / "catalog.json"))
    for uid, u in cat["units"].items():
        if u["stem"] == stem and not u["is_mip"]:
            return uid
    raise SystemExit(f"unknown stem {stem}")


if __name__ == "__main__":
    stem, cand = sys.argv[1], sys.argv[2]
    uid = unit_of(stem)
    orig = ROOT / "pipeline" / "units" / uid / "orig.png"
    m = tol.metrics(orig, cand)
    print(json.dumps(m))
    if len(sys.argv) > 3:
        o, c = Image.open(orig).convert("RGBA"), Image.open(cand).convert("RGBA")
        W, H = min(512, c.width), int(min(512, c.width) * c.height / c.width)
        oo, cc = o.resize((W, H), Image.NEAREST if o.width * 4 <= W else Image.LANCZOS), c.resize((W, H), Image.LANCZOS)
        def on(im, bg):
            b = Image.new("RGBA", im.size, bg); b.alpha_composite(im); return b.convert("RGB")
        def chk(im):
            yy, xx = np.mgrid[0:H, 0:W]; g = (((xx // 16) + (yy // 16)) % 2 * 60 + 110).astype(np.uint8)
            b = Image.fromarray(np.dstack([g, g, g])).convert("RGBA"); b.alpha_composite(im); return b.convert("RGB")
        ao, ac = np.asarray(oo.getchannel("A"), np.int16), np.asarray(cc.getchannel("A"), np.int16)
        tiles = [chk(oo), on(oo, (20, 20, 24, 255)), chk(cc), on(cc, (20, 20, 24, 255)),
                 Image.fromarray(ao.astype(np.uint8)).convert("RGB"), Image.fromarray(ac.astype(np.uint8)).convert("RGB"), Image.fromarray(np.abs(ao - ac).astype(np.uint8)).convert("RGB")]
        sh = Image.new("RGB", (W * 4 + 30, H * 2 + 10), (40, 40, 40))
        for i, t in enumerate(tiles):
            sh.paste(t, ((i % 4) * (W + 10), (i // 4) * (H + 10)))
        sh.save(sys.argv[3])
    sys.exit(0 if m["pass"] else 1)
