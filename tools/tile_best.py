"""tile_best.py NAME TILE [--n 8] [--seed0 300]: redraw one tile with N seeds and keep the draw whose lettering OCR-reads closest to the text its prompt quotes.
Expected words = the single-quoted strings in the tile's prompt (prompts.json) of 4+ letters; score = mean best fuzzy match of each expected word among the OCR words of the tile,
minus a small penalty per confident OCR word that is in nothing quoted (invented text). The best draw is left as out_YX.png (the previous one is kept as out_YX.prev.png).
Reassemble the stage afterwards (tile_job.py assemble, or the job's seam_assemble.py)."""
import argparse, difflib, json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tile_job as tj


def words(s):
    return [w for w in re.findall(r"[A-Za-z0-9]+", s) if len(w) >= 4 or w.isdigit() and len(w) >= 2]


def expected(name, tile):
    p = tj.prompts(name).get(tile, "")
    segs = [s for s in re.findall(r"'([^']{4,200})'", p)]
    out = []
    for s in segs:
        out += words(s)
    return out


def ocr_tokens(img):
    env = dict(os.environ, TESSDATA_PREFIX=str(Path.home() / ".local/share/tessdata"))
    toks = []
    g = img.convert("L")
    for inv in (False, True):
        im2 = g.point(lambda v: 255 - v) if inv else g
        with tempfile.TemporaryDirectory() as td:
            pth = Path(td) / "t.png"; im2.save(pth)
            r = subprocess.run(["tesseract", str(pth), "stdout", "--psm", "11", "tsv"], capture_output=True, text=True, timeout=120, env=env).stdout
        for ln in r.splitlines()[1:]:
            f = ln.split("\t")
            if len(f) >= 12 and f[11].strip():
                try:
                    c = float(f[10])
                except ValueError:
                    continue
                for w in re.findall(r"[A-Za-z0-9]+", f[11]):
                    if len(w) >= 3:
                        toks.append((w.lower(), c))
    return toks


def score(img, exp):
    toks = ocr_tokens(img)
    tw = [t for t, _ in toks]
    if not exp:
        return 0.0, 0
    hit = sum(max([difflib.SequenceMatcher(None, e.lower(), t).ratio() for t in tw] + [0]) for e in exp) / len(exp)
    elow = [e.lower() for e in exp]
    extra = sum(1 for t, c in toks if c >= 70 and len(t) >= 4 and max(difflib.SequenceMatcher(None, t, e).ratio() for e in elow) < 0.7)
    return hit - 0.02 * min(extra, 10), extra


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("tile"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--seed0", type=int, default=300)
    a = ap.parse_args()
    d = tj.jd(a.name) / a.tile.split("_")[0]; yx = a.tile.split("_")[1]
    exp = expected(a.name, a.tile)
    cur = d / f"out_{yx}.png"
    best = (score(Image.open(cur).convert("RGB"), exp)[0], cur.read_bytes()) if cur.exists() else (-9, None)
    print(f"{a.tile}: {len(exp)} expected words; current draw scores {best[0]:.3f}", flush=True)
    shutil.copy(cur, d / f"out_{yx}.prev.png") if cur.exists() else None
    for k in range(a.n):
        tj.gen(a.name, a.tile, a.seed0 + k)
        shutil.copy(d / f"out_{yx}.png", d / f"out_{yx}.s{a.seed0 + k}.png")           # every draw is kept: tile_stitch.py can combine words from different seeds
        s, extra = score(Image.open(d / f"out_{yx}.png").convert("RGB"), exp)
        print(f"  seed {a.seed0 + k}: score {s:.3f} (extra words {extra})", flush=True)
        if s > best[0]:
            best = (s, (d / f"out_{yx}.png").read_bytes())
    (d / f"out_{yx}.png").write_bytes(best[1])
    print(f"{a.tile}: kept best, score {best[0]:.3f}")
