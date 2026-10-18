"""submit_fix.py STEM CAND.png [--force] [--label TEXT]: add a candidate as the newest custom version of the unit (it then shows first in the swipe review, even if the unit was
approved before). Refuses a candidate outside the alpha tolerance unless --force. Uses read-modify-write on units/<uid>/custom/custom.json only."""
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
from PIL import Image
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tol, alpha_tol
ap = argparse.ArgumentParser(); ap.add_argument("stem"); ap.add_argument("cand"); ap.add_argument("--force", action="store_true"); ap.add_argument("--label", default="FLUX redo, shape-guided")
a = ap.parse_args()
uid = alpha_tol.unit_of(a.stem); U = HERE.parent / "pipeline" / "units" / uid
m = tol.metrics(U / "orig.png", a.cand)
if not m["pass"] and not a.force:
    print("outside tolerance, not submitted:", json.dumps(m)); sys.exit(1)
cd = U / "custom"; cd.mkdir(exist_ok=True)
jp = cd / "custom.json"; items = json.loads(jp.read_text()).get("items", []) if jp.exists() else []
h = hashlib.sha1(Path(a.cand).read_bytes()).hexdigest()
if any(i.get("sha1") == h for i in items):
    print("already submitted"); sys.exit(0)
n = max([int(i["id"].split(":")[1]) for i in items] + [0]) + 1
shutil.copy(a.cand, cd / f"up{n}.png")
im = Image.open(a.cand).convert("RGBA")
items.append({"id": f"custom:{n}", "file": f"up{n}.png", "rgba": bool(np.asarray(im.getchannel("A")).min() < 250), "alpha": "FLUX redraw, alpha checked against the original", "size": list(im.size),
              "ts": time.time(), "replaces": "klein:1", "sha1": h, "label": a.label})
jp.write_text(json.dumps({"items": items}, indent=2))
hp = HERE.parent / "pipeline" / "hidden.json"                                    # a unit held out of the review while it is being redone comes back with its new version
try:
    hd = json.loads(hp.read_text())
    if uid in hd.get("uids", []):
        hd["uids"] = [u for u in hd["uids"] if u != uid]; hp.write_text(json.dumps(hd, indent=1))
except (OSError, ValueError):
    pass
print(f"submitted as custom:{n}", json.dumps(m))
