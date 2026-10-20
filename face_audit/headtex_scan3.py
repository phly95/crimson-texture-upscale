"""headtex_scan3.py SCAN2.json OUT.json: for every candidate in scan2.json, score how much structure sits next to each collapsed triangle's UV line.
perp = mean |colour at the line - colour 1..6 texels to the side| (0..255): ~0 means the strip lies inside a uniform band (a deliberate one-texel gradient column), large means the strip picks one column out of detailed art (a real wrap defect)."""
import sys, json
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_scan as SC, headtex_scan2 as S2, headtex_figure as FG
def score(tex, T, bad):
    h, w = tex.shape[:2]; out = []; f = tex.astype(np.float32)
    for j in bad:
        u = np.array(T[j][1]) * [w, h]; e = [(u[i], u[(i + 1) % 3]) for i in range(3)]; a, b = max(e, key=lambda p: np.linalg.norm(p[0] - p[1])); L = np.linalg.norm(b - a)
        if L < 1e-6: continue
        d = (b - a) / L; n = np.array([-d[1], d[0]]); ts = np.linspace(0, 1, max(int(L), 2)); pts = a[None] + ts[:, None] * (b - a)[None]
        def samp(p): return cv2.remap(f, np.mod(p[:, 0] - .5, w).astype(np.float32)[None], np.mod(p[:, 1] - .5, h).astype(np.float32)[None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)[0]
        c0 = samp(pts); dv = np.mean([np.abs(samp(pts + n[None] * k) - c0).mean() for k in (-6, -3, -1, 1, 3, 6)]); out.append(float(dv))
    return float(np.median(out)) if out else 0.0
if __name__ == "__main__":
    res = json.load(open(sys.argv[1])); td = S2.SafeTD()
    for x in res:
        models, mats, tn = S2.pack(x["pack"]); V, T = SC.model_tris(models[x["model"]], mats, tn); rel, a3 = SC.analyse(V, T)
        js = [j for j, t in enumerate(T) if t[2] == x["stem"]]; w, h = SC._CAT.get(x["stem"], (128, 128)); bad = S2.sliver_idx(T, js, a3, w, h)
        x["perp"] = round(score(td[x["stem"]], T, bad), 1)
    json.dump(res, open(sys.argv[2], "w")); print(len(res))
