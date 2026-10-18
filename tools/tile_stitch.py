"""tile_stitch.py NAME TILE: combine words from several draws of one tile. Different seeds garble different words, so a word the base draw got wrong is replaced by the same word
from another draw of that tile (out_YX.s<seed>.png, kept by tile_best.py) that OCR-reads it exactly.
Per word: find the base draw's OCR token that is a near miss of an expected word (the single-quoted strings of the tile's prompt), look in the other draws for a token at the same place that
equals the expected word, register that draw's crop to the base (small translation by cross-correlation), paste it with a feathered mask and the base's local tone. Writes out_YX.png
(old one kept as out_YX.pre_stitch.png) and reports exact-word counts before/after."""
import difflib, glob, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tile_job as tj
import tile_best as tb


def tokens(img):
    """[(text, (x0, y0, x1, y1), conf)] OCR words of 3+ letters, both polarities"""
    env = dict(os.environ, TESSDATA_PREFIX=str(Path.home() / ".local/share/tessdata"))
    g = img.convert("L"); out = []
    for inv in (False, True):
        im2 = g.point(lambda v: 255 - v) if inv else g
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.png"; im2.save(p)
            r = subprocess.run(["tesseract", str(p), "stdout", "--psm", "11", "tsv"], capture_output=True, text=True, timeout=120, env=env).stdout
        for ln in r.splitlines()[1:]:
            f = ln.split("\t")
            if len(f) >= 12 and f[11].strip():
                try:
                    c = float(f[10]); x, y, w, h = int(f[6]), int(f[7]), int(f[8]), int(f[9])
                except ValueError:
                    continue
                t = re.sub(r"[^A-Za-z0-9]", "", f[11])
                if len(t) >= 3 and c >= 30:
                    out.append((t, (x, y, x + w, y + h), c))
    return out


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def register(base_g, other_g, box, pad=8, search=6):
    """integer translation (dx, dy) that best aligns other's crop to base's crop around box"""
    H, W = base_g.shape
    x0, y0, x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad), min(W, box[2] + pad), min(H, box[3] + pad)
    a = base_g[y0:y1, x0:x1]
    best, bd = None, (0, 0)
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            xa, ya = x0 + dx, y0 + dy
            if xa < 0 or ya < 0 or xa + (x1 - x0) > W or ya + (y1 - y0) > H:
                continue
            b = other_g[ya:ya + (y1 - y0), xa:xa + (x1 - x0)]
            e = float(((a - b) ** 2).mean())
            if best is None or e < best:
                best, bd = e, (dx, dy)
    return bd


def stitch(name, tile, log=print):
    d = tj.jd(name) / tile.split("_")[0]; yx = tile.split("_")[1]
    basep = d / f"out_{yx}.png"
    base = Image.open(basep).convert("RGB")
    exp = {w.lower(): w for w in tb.expected(name, tile)}
    draws = sorted(glob.glob(str(d / f"out_{yx}.s*.png")))
    if not exp or not draws:
        log("nothing to do"); return 0
    btoks = tokens(base)
    others = [(p, Image.open(p).convert("RGB")) for p in draws]
    otoks = {p: tokens(im) for p, im in others}

    def exact(toks):
        return sum(1 for t, _, _ in toks if t.lower() in exp)
    before = exact(btoks)
    bg = np.asarray(base.convert("L"), np.float32); arr = np.asarray(base).astype(np.float32).copy()
    fixed = 0
    for t, box, c in btoks:
        if t.lower() in exp:
            continue
        cand = max(exp, key=lambda e: difflib.SequenceMatcher(None, t.lower(), e).ratio())
        if difflib.SequenceMatcher(None, t.lower(), cand).ratio() < 0.6:
            continue                                                   # not a near miss of an expected word: leave it
        bc = center(box); h = box[3] - box[1]
        pick = None
        for p, im in others:
            for ot, ob, oc in otoks[p]:
                if ot.lower() == cand and np.hypot(*(np.array(center(ob)) - np.array(bc))) <= max(14, 0.6 * h) and (pick is None or oc > pick[3]):
                    pick = (p, im, ob, oc)
        if not pick:
            continue
        p, im, ob, oc = pick
        og = np.asarray(im.convert("L"), np.float32)
        union = (min(box[0], ob[0]) - 4, min(box[1], ob[1]) - 4, max(box[2], ob[2]) + 4, max(box[3], ob[3]) + 4)
        dx, dy = register(bg, og, union)
        X0, Y0, X1, Y1 = max(6, union[0]), max(6, union[1]), min(base.size[0] - 6, union[2]), min(base.size[1] - 6, union[3])
        src = np.asarray(im).astype(np.float32)[Y0 + dy:Y1 + dy, X0 + dx:X1 + dx]
        if src.shape[:2] != (Y1 - Y0, X1 - X0):
            continue
        ring = np.ones(src.shape[:2], bool); ring[3:-3, 3:-3] = False
        offs = arr[Y0:Y1, X0:X1][ring].mean(0) - src[ring].mean(0)        # match the local tone at the border
        src = src + offs
        m = np.zeros(src.shape[:2], np.float32); m[3:-3, 3:-3] = 1
        m = cv2.GaussianBlur(m, (0, 0), 2.0)[..., None]
        arr[Y0:Y1, X0:X1] = arr[Y0:Y1, X0:X1] * (1 - m) + src * m
        fixed += 1; log(f"  '{t}' -> '{exp[cand]}' from {Path(p).name} (shift {dx},{dy})")
    out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    shutil.copy(basep, d / f"out_{yx}.pre_stitch.png")
    out.save(basep)
    log(f"{tile}: exact expected words before {before}, after {exact(tokens(out))} ({fixed} words replaced)")
    return fixed


if __name__ == "__main__":
    stitch(sys.argv[1], sys.argv[2])
