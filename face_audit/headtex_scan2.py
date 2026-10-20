"""headtex_scan2.py OUTDIR: visual check of the wrap-defect candidates in headtex_scan.json (known=False, broken>0).
For every candidate texture: pick the model with the most collapsed-UV triangles, render it with the real (original) textures from 8 yaws, once plain and once with the collapsed triangles painted magenta;
the share of the model's pixels that is magenta (best view) ranks how visible the defect is.  Writes OUTDIR/<stem>.png (plain|flagged of the two worst views) and OUTDIR/scan2.json."""
import sys, json, collections, multiprocessing as mp
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_scan as SC, headtex_render as HR, headtex_figure as FG
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else None
YAWS = (0, 45, 90, 135, 180, 225, 270, 315); S = 288; BG = (40, 40, 40); MAG = np.array([255, 0, 255], np.uint8)
_P = {}
def pack(p):
    if p not in _P: _P[p] = SC.load(p)
    return _P[p]
def sliver_idx(T, js, a3, w, h):
    med = np.median(a3[js][a3[js] > 1e-9]) if (a3[js] > 1e-9).any() else 0; out = []
    for j in js:
        u = np.array(T[j][1]) * [w, h]; e = [np.linalg.norm(u[i] - u[(i + 1) % 3]) for i in range(3)]; area = abs((u[1][0] - u[0][0]) * (u[2][1] - u[0][1]) - (u[1][1] - u[0][1]) * (u[2][0] - u[0][0])) / 2
        if max(e) >= 4 and 2 * area / max(e) < 0.75 and a3[j] >= 0.1 * med > 0: out.append(j)
    return out
def candidates():
    return [r["stem"] for r in json.load(open(SC.ROOT / "face_audit/headtex_scan.json")) if r["broken"] > 0 and not r["known"]]
def best_models(stems):
    """stem -> (pack, model index, n slivers) over all packs (one pass)"""
    want = set(stems); best = {}; cat = SC._CAT
    for pk in ["planes.zip"] + [f"{c}/gamez.zip" for c in sorted(p.name for p in SC.EX.iterdir() if (p / "gamez.zip").exists())]:
        models, mats, tn = SC.load(pk)
        for mi, m in enumerate(models):
            V, T = SC.model_tris(m, mats, tn)
            if not T: continue
            by = collections.defaultdict(list)
            for j, t in enumerate(T):
                if t[2] in want: by[t[2]].append(j)
            if not by: continue
            rel, a3 = SC.analyse(V, T)
            for nm, js in by.items():
                w, h = cat.get(nm, (128, 128)); n = len(sliver_idx(T, js, a3, w, h))
                if n > best.get(nm, (0, 0, 0, 0))[2]: best[nm] = (pk, mi, n, len(js))
    return best
def work(a):
    stem, (pk, mi, n, nt) = a
    models, mats, tn = pack(pk); V, T = SC.model_tris(models[mi], mats, tn); rel, a3 = SC.analyse(V, T)
    js = [j for j, t in enumerate(T) if t[2] == stem]; w, h = SC._CAT.get(stem, (128, 128)); bad = set(sliver_idx(T, js, a3, w, h))
    T2 = [(vi, uv, "__sliver" if j in bad else nm) for j, (vi, uv, nm) in enumerate(T)]
    HR._M[("s2:" + pk, mi)] = (V, T); HR._M[("s2f:" + pk, mi)] = (V, T2)
    td = SafeTD()
    plain, flg, frac = [], [], []
    for y in YAWS:
        a_ = HR.render(td, y, 0, S=S, flipv=False, model=mi, pack="s2:" + pk, engine=True, bg=BG)
        f_ = HR.render(_Merge(td), y, 0, S=S, flipv=False, model=mi, pack="s2f:" + pk, engine=True, bg=BG)
        mod = (np.abs(a_.astype(int) - BG).max(-1) > 6).sum(); mg = ((f_ == MAG).all(-1)).sum()
        plain.append(a_); flg.append(f_); frac.append(float(mg / max(mod, 1)))
    order = np.argsort(frac)[::-1][:2]
    tile = np.vstack([np.hstack([plain[i] for i in order]), np.hstack([flg[i] for i in order])])
    if OUT: Image.fromarray(tile).save(OUT / f"{stem}.png")
    return dict(stem=stem, pack=pk, model=mi, slivers=n, tris=nt, verts=len(V), frac=max(frac), fracmean=float(np.mean(frac)), views=[int(YAWS[i]) for i in order])
class SafeTD(FG.TexDict):
    def __missing__(self, k):
        try: return super().__missing__(k)
        except OSError: self[k] = np.full((8, 8, 3), 128, np.uint8); return self[k]
class _Merge(dict):
    """originals via FG.TexDict plus the magenta flag texture"""
    def __init__(self, td): super().__init__(); self.td = td
    def __missing__(self, k): return np.full((8, 8, 3), MAG, np.uint8) if k == "__sliver" else self.td[k]
if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); st = candidates(); print(len(st), "candidates", flush=True); b = best_models(st); print("models picked", len(b), flush=True)
    with mp.Pool(4) as pool: res = pool.map(work, list(b.items()), chunksize=2)
    res.sort(key=lambda r: -r["frac"]); json.dump(res, open(OUT / "scan2.json", "w"), indent=0); print("done", len(res))
