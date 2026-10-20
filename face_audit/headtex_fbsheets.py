"""headtex_fbsheets.py TEX [SUIT01 SUIT02]: review renders for fbhead1: head sheet (original vs new, 6 angles), large 3/4 + profile, figure sheet"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_figure as FG, headtex_render as HR
D = FG.ROOT / "face_audit/headtex"; L = lambda p: np.array(Image.open(p).convert("RGB"))
def unit_file(stem, f): return FG.ROOT / "pipeline/units" / next(u for u, v in FG.cat.items() if v["stem"] == stem and not v.get("is_mip")) / f
t = L(sys.argv[1]); new = {"fbhead1": t}
appr = {s: L(unit_file(s, "klein_s2_cm.png")) for s in ("fbsuit128_01", "fbsuit128_02")}
if len(sys.argv) > 3: new.update({"fbsuit128_01": L(sys.argv[2]), "fbsuit128_02": L(sys.argv[3])})
FG.sheet([("original", FG.texdict()), ("new cartoon fbhead1", FG.texdict({"fbhead1": t}))], 1124, "C1C/gamez.zip", D / "fbhead1_sheet.png", S=420)
tt = FG.texdict({"fbhead1": t}); Image.fromarray(np.hstack([HR.render(tt, y, S=800, flipv=False, model=1124, pack="C1C/gamez.zip", engine=True) for y in (215, 270)])).save(D / "fbhead1_big.png")
k, p = FG.fig("fbman"); rows = [("original", FG.texdict()), ("new head + approved suit (klein:2)", FG.texdict({"fbhead1": t, **appr}))]
if len(sys.argv) > 3: rows.append(("new head + cartoon suit candidates", FG.texdict(new)))
FG.sheet(rows, k, p, D / "fbhead1_figure.png", S=420)
