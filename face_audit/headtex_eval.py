"""headtex_eval.py TARGET [front_file]: run the hard gates of headtex_criteria.md on a target's redrawn views and on the baked texture re-rendered on the mesh"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_score as S, headtex_run as R, headtex_bake as B, headtex_render as HR
D = R.D
def evaluate(tg, front=None, tex_file=None):
    C = R.TARGETS[tg]; rows = []; ok = True
    for k, (y, p) in R.VIEWS.items():
        ref = np.array(Image.open(D / f"v_{tg}_{k}.png").convert("RGB")); f = D / (front if (k == "front" and front) else f"r_{tg}_{k}.png"); cand = np.array(Image.open(f).convert("RGB"))
        if k in ("back", "top"):
            fc = S.face(cand); fl = [] if (fc is None or fc["score"] < 0.5) else ["face hallucinated (score %.2f)" % fc["score"]]; m = dict(lowcorr=S.lowcorr(ref, cand))
            if m["lowcorr"] < S.THR["lowcorr"]: fl.append("layout drift %.3f" % m["lowcorr"])
        else: m, fl = S.score_view(ref, cand, k == "front", f"{tg}_{k}")
        rows.append((k, "redraw", m, fl))
    if tex_file:                                                                         # round trip: the baked texture drawn on the mesh again (UVs of the two collapsed triangles corrected, as in the references)
        tex = np.array(Image.open(tex_file).convert("RGB")); tx = {n: tex for n in {t[2] for t in HR.tris(C["model"], C["pack"])[1] if t[2]}} if len(C["bake"]) == 1 else None
        for k, (y, p) in R.VIEWS.items():
            ref = np.array(Image.open(D / f"v_{tg}_{k}.png").convert("RGB")); im, zb = B.render_view(tx, y, p, model=C["model"], pack=C["pack"]); mask = zb > -1e8
            m = dict(lowcorr=S.lowcorr(np.where(mask[..., None], ref, 128).astype(np.uint8), np.where(mask[..., None], im, 128).astype(np.uint8)))
            fl = [] if m["lowcorr"] >= 0.90 else ["round-trip drift %.3f" % m["lowcorr"]]; rows.append((k, "baked on mesh", m, fl))
    for k, what, m, fl in rows:
        ok &= not fl; print("%-6s %-14s %s  %s" % (k, what, {a: round(b, 3) for a, b in m.items() if isinstance(b, float)}, "PASS" if not fl else "FAIL: " + "; ".join(fl)))
    print("OVERALL", "PASS" if ok else "FAIL"); return ok
if __name__ == "__main__": evaluate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None, sys.argv[3] if len(sys.argv) > 3 else None)
