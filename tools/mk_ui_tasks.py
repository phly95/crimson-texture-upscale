"""mk_ui_tasks.py: turn the game's menu / fullscreen / scrapbook art into pipeline tasks (tasks/tui*/task.json + orig/<uid>.png), same format the
texture units use, so tools/pipeline/run.py redraws and the review page shows them like any other unit.  Also writes ui_catalog.json (uid -> source file in
the game's extracted/ tree) for the pack builder.  Skipped on purpose: *_MASK.PNG and the per-plane-pattern folders (derived dumps of the .BM skins), PX_ICON_* (solid
white or white-on-alpha silhouettes), the bitmap fonts (FONT/ARIAL8: glyph metrics), and sprite atlases more than 4:1."""
import glob, hashlib, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/home/deck/workspace/csvm-hd")
X = Path.home() / "Games/CSVM/extracted"
files = []
for pat, tag, note in (("rimage/*.png", "rimage", "menu art from the game's mission-select and briefing screens (photo prints, maps, portraits, HUD and screen pieces)"),
                       ("rof/ASSETS/GRAPHICS/*.*", "gfx", "menu / interface art of the game (button strips, panels, backgrounds)"),
                       ("rof/ASSETS/GRAPHICS/SCRAPBOOK/*.*", "scrap", "scrapbook / briefing prop art of the game (1930s photographs, newspaper clippings, posters, blueprints)")):
    for f in sorted(glob.glob(str(X / pat))):
        p = Path(f); n = p.name.upper()
        if p.suffix.lower() not in (".png", ".jpg", ".tga", ".tif") or "_MASK." in n or n.startswith("PX_ICON") or n in ("FONT.TGA", "ARIAL8.TGA"):
            continue
        files.append((p, tag, note))
units, skipped = [], []
for p, tag, note in files:
    try:
        im = Image.open(p); im.load()
    except Exception as e:                                       # noqa: BLE001
        skipped.append((str(p), repr(e)[:60])); continue
    w, h = im.size
    if max(w, h) / min(w, h) > 4 or min(w, h) < 8:
        skipped.append((str(p.relative_to(X)), f"atlas/strip {w}x{h}")); continue
    rgba = im.convert("RGBA")
    a = np.asarray(rgba.getchannel("A"))
    has_alpha = bool(a.min() < 255)
    sha = hashlib.sha1(p.read_bytes()).hexdigest()
    stem = f"{tag}_{p.stem}"
    uid = f"{stem}__{sha[:8]}"
    sc = min(8.0, 2048 / max(w, h))
    rgb = np.asarray(rgba.convert("RGB")).astype(np.float32)
    n = len({tuple(c) for c in (rgb.reshape(-1, 3)[::7] // 16).astype(int)})
    units.append({"uid": uid, "stem": stem, "size": [w, h], "target": [round(w * sc), round(h * sc)], "has_alpha": has_alpha, "alpha_class": "Full" if has_alpha else "None",
                  "alpha_is_soft": bool(has_alpha and ((a[a > 0] >= 200).mean() < 0.45)), "tileable_guess": False, "seam_ratio": 9.9,
                  "mean_rgb": [round(float(x), 1) for x in rgb.reshape(-1, 3).mean(0)], "distinct_colours_4bit": n,
                  "look": "flat-graphic-likely" if n <= 40 else "photo/painted-likely", "also_named": [], "chapters": [], "flipbook_id": None, "usage": {},
                  "ui_note": note, "ui_src": str(p.relative_to(X)), "ui_ext": p.suffix.lower(), "ui_dir": str(p.parent.relative_to(X))})
    d = ROOT / "tasks" / f"tui{(len(units) - 1) // 24:04d}" / "orig"
    d.mkdir(parents=True, exist_ok=True)
    rgba.save(d / f"{uid}.png")
by = {}
for u in units:
    by.setdefault(f"tui{(units.index(u)) // 24:04d}", []).append(u)
for tid, us in by.items():
    json.dump({"task": tid, "kind": "ui", "family": "ui", "units": us}, open(ROOT / "tasks" / tid / "task.json", "w"), indent=1)
json.dump({"units": {u["uid"]: {k: u[k] for k in ("stem", "size", "target", "has_alpha", "ui_src", "ui_ext")} for u in units}, "skipped": skipped}, open(ROOT / "ui_catalog.json", "w"), indent=1)
print(len(units), "units in", len(by), "tasks;", len(skipped), "skipped")
for s in skipped[:40]: print("  skip", s)
