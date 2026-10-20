"""headtex_fbcheck.py: renders of the original fbhead1 head + full fbman figure (original textures)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_figure as FG, headtex_render as HR
D = FG.ROOT / "face_audit/headtex"
t = FG.texdict()
FG.sheet([("orig head", t)], 1124, "C1C/gamez.zip", D / "fb_orig_head.png", S=420)
k, p = FG.fig("fbman"); FG.sheet([("orig figure", t)], k, p, D / "fb_orig_fig.png", S=420)
V, T = HR.tris(1124, "C1C/gamez.zip"); d = HR.density(V, T)
for i, (vi, uv, nm) in enumerate(T): print(i, nm, "%.3f" % d[i], [tuple(round(x * 128, 1) for x in u) for u in uv])
