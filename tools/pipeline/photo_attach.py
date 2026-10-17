"""photo_attach.py UID IMAGE --title T --url U --license L [--author A] [--crop x0,y0,x1,y1] [--orient auto]

Adds a REAL PHOTO as another version of a finished unit (for things like the moon where a proper photo is easy to find): fits the photo to the unit's frame
(crop to the frame's aspect, resize), optionally searches the orientation (8 flips/rotations x 10-degree steps) that best matches the original, colour-matches its
broad tone to the original (colormatch), stores it in units/<uid>/photo/ with its source and licence in meta.json, and puts the unit back in the review queue.
Only use public-domain / CC0 sources (NASA, Wikimedia Commons PD, Poly Haven, ambientCG)."""
import argparse, json, os, shutil, sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from colormatch import colormatch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("uid"); ap.add_argument("image")
ap.add_argument("--title", required=True); ap.add_argument("--url", required=True); ap.add_argument("--license", required=True); ap.add_argument("--author", default="")
ap.add_argument("--crop"); ap.add_argument("--orient", default="none")
a = ap.parse_args()

D = ROOT / "pipeline" / "units" / a.uid
if not (D / "meta.json").exists():
    sys.exit("no such unit")
W, H = Image.open(D / "in.png").size
im = Image.open(a.image).convert("RGB")
if a.crop:
    x0, y0, x1, y1 = (int(v) for v in a.crop.split(","))
    im = im.crop((x0, y0, x1, y1))
ar = W / H
w, h = im.size
if w / h > ar:                                           # centre-crop to the frame's aspect
    nw = int(h * ar); im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
else:
    nh = int(w / ar); im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
im = im.resize((W, H), Image.LANCZOS)

orient = None
if a.orient == "auto":
    o = np.asarray(Image.open(D / "orig.png").convert("L").resize((64, 64), Image.BOX)).astype(np.float32)
    best = (-9, None)
    for flip in (False, True):
        base = im.transpose(Image.FLIP_LEFT_RIGHT) if flip else im
        for ang in range(0, 360, 10):
            c = np.asarray(base.rotate(ang, Image.BICUBIC).convert("L").resize((64, 64), Image.BOX)).astype(np.float32)
            r = float(np.corrcoef(o.ravel(), c.ravel())[0, 1])
            if r > best[0]:
                best = (r, (flip, ang))
    (flip, ang) = best[1]; orient = {"flip": flip, "rotate": ang, "corr": round(best[0], 3)}
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = im.rotate(ang, Image.BICUBIC)

P = D / "photo"; P.mkdir(exist_ok=True)
im.save(P / "klein_s1.png")
shutil.copy(D / "orig.png", P / "orig.png")
import colormatch as _cm
_cm.CLIP = np.array([45.0, 32.0, 32.0], np.float32)             # a different photograph needs a bigger tone shift than a redraw of the same picture
colormatch(P, 1, strength=1.0)
src = {"title": a.title, "url": a.url, "license": a.license, "author": a.author}
(P / "source.json").write_text(json.dumps(src, indent=1))
mp = D / "meta.json"
m = json.loads(mp.read_text())
m["photo"] = {"status": "ready", "source": src, "orient": orient, "t": time.time()}
tmp = D / "meta.json.tmp"; tmp.write_text(json.dumps(m)); os.replace(tmp, mp)
with open(ROOT / "pipeline" / "reviews.jsonl", "a") as f:                      # back into the review queue
    f.write(json.dumps({"uid": a.uid, "decision": "undo", "chosen": None, "seen": [], "issues": [], "text": "photo ready", "ts": time.time()}) + "\n")
print("ok", orient)
