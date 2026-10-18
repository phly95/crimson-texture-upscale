"""sprite_tool.py: the agent's self-check harness. Run from this directory with  /home/deck/workspace/csvm-hd/venv/bin/python sprite_tool.py ...

  list                 every sprite with its state (todo / attempted / PASSED / SUBMITTED), size, target size
  check STEM [FILE]    verify FILE (default sprites/STEM/draw.png) against sprites/STEM/orig.png; writes sprites/STEM/sheet.png (contact sheet) and,
                       for tile-type sprites, sprites/STEM/tile2x2.png; prints metrics and what to fix. Exit 0 = every automatic gate passes.
  submit STEM [FILE]   re-runs check; only a passing file of the target size is copied to out/STEM.png and recorded as SUBMITTED
  status               same as list plus a one-line count

Gates are computed by sprite_verify.py on the candidate reduced (BOX) to the original's size. Passing the gates is necessary, never sufficient:
the sheet must also LOOK right (same kind of thing as the original, richer detail). Do not edit orig.png, sprite_verify.py or this tool.
"""
import hashlib, json, subprocess, sys, shutil, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import fringe

HERE = Path(__file__).resolve().parent
VERIFY = HERE.parent / "tools" / "agents" / "sprite_verify.py"   # shared with sprite_agent/
PY = sys.executable
STATE = HERE / "state.json"


def sha(p):
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=1))


def seam(img):
    """wrap-around discontinuity / typical neighbour difference, premultiplied RGBA, averaged over both axes (about 1 = seamless, >>1 = visible seam)"""
    a = np.asarray(img.convert("RGBA"), np.float32) / 255
    a = np.dstack([a[..., :3] * a[..., 3:], a[..., 3:]])
    r = []
    for ax in (0, 1):
        edge = np.abs(np.take(a, 0, ax) - np.take(a, -1, ax)).mean()
        inner = np.abs(np.diff(a, axis=ax)).mean()
        r.append(edge / max(inner, 1e-4))
    return float(np.mean(r))


def check(stem, file=None, quiet=False):
    d = HERE / "sprites" / stem
    meta = json.loads((d / "meta.json").read_text())
    f = Path(file) if file else d / "draw.png"
    if not f.exists():
        print(f"{f} does not exist: draw it first"); return False
    guard = json.loads((HERE / "guard.json").read_text())
    if sha(d / "orig.png") != guard["orig"][stem] or sha(VERIFY) != guard["verify"]:
        print("orig.png or sprite_verify.py was modified: refusing"); return False
    r = subprocess.run([PY, "-I", str(VERIFY), str(d / "orig.png"), str(f), str(d / "sheet.png")], capture_output=True, text=True)
    m = json.loads(r.stdout.strip().splitlines()[-1])
    cand = Image.open(f)
    o = Image.open(d / "orig.png")
    tw, th = meta["target"]
    fails = []
    if cand.size != (tw, th):
        fails.append(f"size {cand.size} != target {(tw, th)} (draw at exactly the target size)")
    if m["alpha_mae"] >= .04: fails.append(f"alpha_mae {m['alpha_mae']} >= .04 (alpha differs from the original after reduction)")
    if m["alpha_box32"] >= .06: fails.append(f"alpha_box32 {m['alpha_box32']} >= .06")
    if abs(m["coverage_orig"] - m["coverage_cand"]) >= .05: fails.append(f"coverage {m['coverage_cand']} vs original {m['coverage_orig']} (must be within .05)")
    iou_min = meta.get("iou_min", .85)
    if m["iou50"] <= iou_min: fails.append(f"silhouette iou50 {m['iou50']} <= {iou_min} (shape is in the wrong place or wrong size)")
    if m["soft_lost"] >= .15: fails.append(f"soft_lost {m['soft_lost']} >= .15 (original's soft edge band missing)")
    de_max = meta.get("de_max", 8)
    if m["de_mean"] >= de_max: fails.append(f"colour de_mean {m['de_mean']} >= {de_max} (palette differs from the original)")
    if m["green_under_clear"]: fails.append("green key colour under alpha 0")
    if meta.get("group", "effects") != "effects":
        rim_de, rim_hue = fringe.rim_shift(cand.convert("RGBA").__array__() / 255 if False else np.asarray(cand.convert("RGBA"), np.float32) / 255, np.asarray(o.convert("RGBA"), np.float32) / 255)
        m["rim_hue"] = round(rim_hue, 1)
        if rim_hue > 10:
            fails.append(f"edge rim colour: the outermost solid pixels along the silhouette are off the original's colour by {rim_hue:.0f} (a*/b* distance, limit 10): a dark olive or green rim "
                         "(keying spill / outline) around the object. The edge must be the same material colour as the interior next to it, anti-aliased, no outline.")
    so, sc = seam(o), seam(cand)
    m["seam_orig"], m["seam_cand"] = round(so, 2), round(sc, 2)
    if so < 1.4:
        t = cand.convert("RGBA")
        sheet = Image.new("RGBA", (t.width * 2, t.height * 2), (70, 70, 80, 255))
        for i in range(2):
            for j in range(2):
                sheet.alpha_composite(t, (i * t.width, j * t.height))
        sheet.convert("RGB").save(d / "tile2x2.png")
        if sc > max(so * 1.5, 1.5): fails.append(f"tile seam {sc:.2f} vs original {so:.2f}: this sprite tiles, make the edges wrap (fx.fbm tile=True, warp wraps)")
        m["tiles"] = True
    ok = not fails
    m["pass"] = ok
    s = state(); e = s.setdefault(stem, {"attempts": 0, "status": "todo"})
    e["attempts"] += 1; e["last"] = m; e["time"] = time.strftime("%H:%M:%S")
    if e["status"] != "SUBMITTED": e["status"] = "PASSED" if ok else "attempted"
    save_state(s)
    if not quiet:
        print(json.dumps(m)); print("sheet:", d / "sheet.png", ("  tile preview: " + str(d / "tile2x2.png")) if m.get("tiles") else "")
        print("PASS (now look at the sheet before submitting)" if ok else "FAIL:\n  - " + "\n  - ".join(fails))
    return ok


def submit(stem, file=None):
    d = HERE / "sprites" / stem
    f = Path(file) if file else d / "draw.png"
    if not check(stem, f, quiet=True):
        print("not submitted: gates fail, run check"); return False
    (HERE / "out").mkdir(exist_ok=True)
    shutil.copy(f, HERE / "out" / f"{stem}.png")
    shutil.copy(f, d / f"submitted_{int(time.time())}.png")
    s = state(); s[stem]["status"] = "SUBMITTED"; s[stem]["submitted"] = time.strftime("%H:%M:%S"); save_state(s)
    print(f"SUBMITTED {stem} -> out/{stem}.png"); return True


def listing():
    s = state(); order = json.loads((HERE / "sprites.json").read_text())["order"]; n = 0
    for st in order:
        e = s.get(st, {"status": "todo", "attempts": 0}); n += e["status"] == "SUBMITTED"
        meta = json.loads((HERE / "sprites" / st / "meta.json").read_text())
        print(f"{st:18} {e['status']:10} attempts={e['attempts']:<3} orig={meta['orig_size']} target={meta['target']}")
    print(f"{n}/{len(order)} submitted")


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "list"
    if c in ("list", "status"): listing()
    elif c == "check": sys.exit(0 if check(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None) else 1)
    elif c == "submit": sys.exit(0 if submit(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None) else 1)
    else: print(__doc__)
