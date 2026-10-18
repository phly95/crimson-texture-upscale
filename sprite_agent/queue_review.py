"""queue_review.py [STEM ...]: put submitted sprites from out/ into the swipe review queue.
Each file becomes the unit's newest custom version (units/<uid>/custom/upN.png + custom.json, shown first on the review page). A unit with a newer
unseen version re-enters the review queue by itself (serve_gallery /api/fresh), even if approved. Idempotent per file content.
No STEM = every file in out/ (not _overview).   /home/deck/workspace/csvm-hd/venv/bin/python queue_review.py ripple"""
import hashlib, json, shutil, sys, time
from pathlib import Path
from PIL import Image
import numpy as np

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent / "pipeline"
meta = json.loads((HERE / "sprites.json").read_text())["sprites"]
stems = sys.argv[1:] or sorted(p.stem for p in (HERE / "out").glob("*.png") if not p.stem.startswith("_"))
last = {}
for ln in (PIPE / "reviews.jsonl").read_text().splitlines():
    try:
        r = json.loads(ln); last[r["uid"]] = r
    except (ValueError, KeyError):
        pass
for s in stems:
    uid = meta[s]["uid"]; src = HERE / "out" / f"{s}.png"; cdir = PIPE / "units" / uid / "custom"
    cdir.mkdir(exist_ok=True); jp = cdir / "custom.json"
    items = json.loads(jp.read_text()).get("items", []) if jp.exists() else []
    h = hashlib.sha1(src.read_bytes()).hexdigest()
    if any(it.get("sha1") == h for it in items):
        print(f"{s}: already queued ({[it['id'] for it in items if it.get('sha1') == h][0]})"); continue
    n = max([int(it["id"].split(":")[1]) for it in items] + [0]) + 1
    shutil.copy(src, cdir / f"up{n}.png")
    im = Image.open(src).convert("RGBA")
    items.append({"id": f"custom:{n}", "file": f"up{n}.png", "rgba": bool(np.asarray(im.getchannel("A")).min() < 250), "alpha": "drawn with explicit alpha (procedural, checked against the original)",
                  "size": list(im.size), "ts": time.time(), "replaces": (last.get(uid) or {}).get("chosen") or "klein:1", "sha1": h,
                  "label": "Procedural redraw (Gemini agent)"})
    jp.write_text(json.dumps({"items": items}, indent=2))
    d = last.get(uid, {})                                     # the existing approval stays: the review page re-queues a unit that has a newer unseen version
    print(f"{s}: queued as custom:{n} (was {d.get('decision')})")
