"""alpha_recipes.py - the alpha ("transparency") recipes behind the review page's Alpha fix button.
  python alpha_recipes.py options UID VERSION   -> writes units/UID/alphafix/<version>/opt_<id>.jpg (320 px, diagonal split: dark / light background) and prints a JSON list [{id,label}]
  python alpha_recipes.py apply UID OPTION      -> applies the option to every version of the unit (first pass and retry, all seeds) from the colour-matched redraws; old files are kept as *_rgba.chromakey.png
VERSION is the page's id (klein:1, retry:2).  Units redrawn on a magenta screen (retry_greenkey/), hand-edited units and uploads are refused: those need a person."""
import sys, os, json, shutil, time, re
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from greenkey import green_key
from bloom_alpha import bloom, alpha_from_orig
from fillholes import fill_holes
from PIL import Image

PIPE = Path("/home/deck/workspace/csvm-hd/pipeline")
UN = PIPE / "units"


def soft_rekey(rgb, orig_path, neutral=False, no_bloom=False, gmax=None, alpha_orig=None, core_only=False):
    """soft green key on the colour-matched redraw, then (optionally) the original's glow / alpha, then no green cast left"""
    arr, G, k0 = green_key(rgb, neutralize=neutral)
    out = alpha_from_orig(arr, orig_path, alpha_orig, core_only=core_only) if alpha_orig else (arr if no_bloom else bloom(arr, orig_path, gmax=gmax))
    f = out.astype(np.float32)
    m = (f[..., 3] < 250) | neutral                                  # neutral subjects: no green cast anywhere
    cap = (np.maximum(f[..., 0], f[..., 2]) + 2) if neutral else (f[..., 0] * (0.9 if gmax else 0.97) + 2)
    f[..., 1] = np.where(m, np.minimum(f[..., 1], cap), f[..., 1])
    return np.clip(f, 0, 255).round().astype(np.uint8)


# id, label, kind, params
RECIPES = [
    ("soft_warm", "Soft glow, warm (fire, flames)", "soft", dict(gmax=0.85)),
    ("soft_glow", "Soft glow, colour kept", "soft", dict()),
    ("soft_white", "Soft glow, white / grey (smoke, flash)", "soft", dict(neutral=True)),
    ("soft_edges", "Soft edges only, colour kept", "soft", dict(no_bloom=True)),
    ("soft_edges_white", "Soft edges only, white / grey", "soft", dict(neutral=True, no_bloom=True)),
    ("orig_alpha", "Use original alpha (redraw colours)", "soft", dict(alpha_orig=1.0)),
    ("orig_flare", "Original's glow and alpha (light flares)", "soft", dict(alpha_orig=1.3, core_only=True)),
    ("orig_white", "Use original alpha, white / grey", "soft", dict(alpha_orig=1.0, neutral=True)),
    ("fill", "Hard edge, enclosed parts filled in", "fill", dict()),
    ("hard", "Original automatic hard key", "hard", dict()),
]
LABELS = {r[0]: r[1] for r in RECIPES}
LABELS["current"] = "Keep as it is now"


def _safe(vid):
    return re.sub(r"[^A-Za-z0-9]+", "_", vid)


def _check(uid):
    d = UN / uid
    if not (d / "meta.json").is_file():
        raise SystemExit("unknown unit")
    if (d / "retry_greenkey").exists():
        raise SystemExit("this one was redrawn on a magenta screen: ask Claude to fix its alpha")
    try:
        if uid in json.loads((PIPE / "hand_edited.json").read_text()):
            raise SystemExit("this one was edited by hand: ask Claude to fix its alpha")
    except OSError:
        pass
    m = json.loads((d / "meta.json").read_text())
    if m.get("soft"):
        raise SystemExit("this one is a smooth enlargement (no key to fix)")
    return d, m


def _versions(d):
    out = []
    for sub in ("", "retry/"):
        for s in (1, 2):
            cm = d / sub / f"klein_s{s}_cm.png"
            if cm.exists() and (d / sub / f"klein_s{s}_rgba.png").exists():
                out.append((sub, s))
    return out


def _hard_path(d, sub, s):
    bk = d / sub / f"klein_s{s}_rgba.chromakey.png"
    return bk if bk.exists() else d / sub / f"klein_s{s}_rgba.png"


def compute(d, sub, s, rid):
    cm = np.asarray(Image.open(d / sub / f"klein_s{s}_cm.png").convert("RGB")).astype(np.float32)
    if rid == "current":
        return np.asarray(Image.open(d / sub / f"klein_s{s}_rgba.png").convert("RGBA"))
    kind, params = next((r[2], r[3]) for r in RECIPES if r[0] == rid)
    if kind == "hard":
        return np.asarray(Image.open(_hard_path(d, sub, s)).convert("RGBA"))
    if kind == "fill":
        hard = np.asarray(Image.open(_hard_path(d, sub, s)).convert("RGBA"))
        return fill_holes(hard, cm.astype(np.uint8))[0]
    return soft_rekey(cm, str(d / "orig.png"), **params)


def preview(arr, size=320):
    im = Image.fromarray(arr, "RGBA").resize((size, size), Image.LANCZOS)
    y, x = np.mgrid[:size, :size]
    bg = np.where(((x + y) < size)[..., None], np.array([25, 25, 35], np.uint8), np.array([205, 210, 220], np.uint8)).astype(np.uint8)
    return Image.alpha_composite(Image.fromarray(bg, "RGB").convert("RGBA"), im).convert("RGB")


def options(uid, vid):
    d, m = _check(uid)
    sub, s = ("retry/", vid.split(":")[1]) if vid.startswith("retry:") else ("", vid.split(":")[1]) if vid.startswith("klein:") else (None, None)
    if sub is None or not s.isdigit() or not (d / sub / f"klein_s{s}_cm.png").exists():
        raise SystemExit("alpha fix needs a redraw version (not an upload or photo)")
    s = int(s)
    out = d / "alphafix" / _safe(vid); out.mkdir(parents=True, exist_ok=True)
    ids = ["current"] + [r[0] for r in RECIPES if not (r[0] == "hard" and not (d / sub / f"klein_s{s}_rgba.chromakey.png").exists())]
    res, seen = [], []
    for rid in ids:
        f = out / f"opt_{rid}.jpg"
        src_t = max((d / sub / f"klein_s{s}_cm.png").stat().st_mtime, (d / sub / f"klein_s{s}_rgba.png").stat().st_mtime)
        if not f.exists() or f.stat().st_mtime < src_t:
            arr = compute(d, sub, s, rid)
            preview(arr).save(f, quality=86)
        res.append({"id": rid, "label": LABELS[rid], "t": int(f.stat().st_mtime)})
    return res


def apply(uid, rid):
    if rid not in LABELS:
        raise SystemExit("unknown option")
    d, m = _check(uid)
    if rid == "current":
        return {"applied": rid, "versions": 0}
    n = 0
    for sub, s in _versions(d):
        f = d / sub / f"klein_s{s}_rgba.png"
        bk = d / sub / f"klein_s{s}_rgba.chromakey.png"
        arr = compute(d, sub, s, rid)
        if not bk.exists():
            shutil.copy(f, bk)
        tmp = f.with_suffix(".tmp.png"); Image.fromarray(arr, "RGBA").save(tmp); os.replace(tmp, f); n += 1
    (d / "alphafix").mkdir(exist_ok=True)
    (d / "alphafix" / "applied.json").write_text(json.dumps({"option": rid, "label": LABELS[rid], "ts": time.time(), "versions": n}))
    return {"applied": rid, "versions": n}


if __name__ == "__main__":
    try:
        if sys.argv[1] == "options":
            print(json.dumps(options(sys.argv[2], sys.argv[3])))
        elif sys.argv[1] == "apply":
            print(json.dumps(apply(sys.argv[2], sys.argv[3])))
    except SystemExit as e:
        print(json.dumps({"error": str(e)})); sys.exit(0)
