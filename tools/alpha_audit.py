"""alpha_audit.py: compare the alpha of every shipped (approved) world texture with its original, the way the CSVM author did in #214.

Both alphas are box-averaged to 32x32; 'diff' is their mean absolute difference (0..1), coverage is the mean alpha. Listed when diff > THRESH.
Writes alpha_audit.json (per uid) and prints a ranked table.   venv/bin/python tools/alpha_audit.py [THRESH]
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import build_pack as bp

THRESH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.12     # the practical tolerance (flux_fix/tol.py): diff <= 0.12 and |coverage change| <= 0.15


def box32(a):
    im = Image.fromarray((a * 255).astype(np.uint8)).resize((32, 32), Image.BOX)
    return np.asarray(im).astype(np.float32) / 255


def alpha_of(path):
    return np.asarray(Image.open(path).convert("RGBA").getchannel("A")).astype(np.float32) / 255


def main():
    cat = json.load(open(bp.ROOT / "catalog.json"))
    decs = bp.decisions()
    rows = []
    for uid, u in cat["units"].items():
        if u["is_mip"] or u["is_placeholder"] or not u["has_alpha"]:
            continue
        r = decs.get(uid, {})
        if r.get("decision") != "approve" or not r.get("chosen"):
            continue
        src = bp.version_file(uid, r["chosen"])
        if src is None:
            continue
        p, _ = bp.conform(uid, src, cat)
        ao, ah = box32(alpha_of(bp.orig_path(u))), box32(alpha_of(p))
        rows.append({"uid": uid, "stem": u["stem"], "chapters": u["chapters"], "chosen": r["chosen"],
                     "diff": float(np.abs(ao - ah).mean()), "orig": float(ao.mean()), "hd": float(ah.mean())})
    rows.sort(key=lambda x: -x["diff"])
    json.dump(rows, open(bp.ROOT / "alpha_audit.json", "w"), indent=1)
    bad = [x for x in rows if x["diff"] > THRESH or abs(x["hd"] - x["orig"]) > 0.15]
    print(f"{len(rows)} approved cutouts, {len(bad)} with diff > {THRESH}")
    for x in bad:
        print(f"{x['stem']:28} {x['diff']:.2f} {x['orig']*100:4.0f}% -> {x['hd']*100:4.0f}%  {x['chosen']:10} {','.join(x['chapters'])}")


main()
