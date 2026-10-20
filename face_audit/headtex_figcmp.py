"""headtex_figcmp.py [S]: three-row full-figure sheets (original game textures, v2.0.2 pack, current hand-drawn; front, 3/4, side, back; corrected mesh) -> OUT/figcmp_<name>.png.  OLD = folder with the v2.0.2 world_textures"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_figure as FG, headtex_render as HR, headtex_figbig as FB
OLD = Path(sys.argv[2]); OUT = Path(sys.argv[3])
def old(stem): return np.array(Image.open(OLD / f"world_textures/{stem}.png").convert("RGB"))
def run(S=1500, yaws=(180, 135, 90, 0)):
    BG = FB.BG
    for name, (fg, tex) in FB.sets().items():
        key, pack = FG.fig(fg, True)
        rows = []
        for t in (FG.texdict(), FG.texdict({k: old(k) for k in tex}), FG.texdict(tex)):
            rows.append([HR.render(t, y, 0, S=S, flipv=False, model=key, pack=pack, engine=True, bg=BG) for y in yaws])
        m = np.zeros((S, S), bool)
        for r in rows:
            for im in r: m |= (np.abs(im.astype(int) - BG).max(-1) > 6)
        ys, xs = np.nonzero(m); y0, y1 = max(ys.min() - 12, 0), min(ys.max() + 12, S); cw = (xs.max() - xs.min()) + 24
        out = []
        for r in rows:
            cols = []
            for im in r:
                mm = (np.abs(im.astype(int) - BG).max(-1) > 6)
                cx = int(np.nonzero(mm.any(0))[0].mean()) if mm.any() else S // 2
                x0 = int(np.clip(cx - cw // 2, 0, S - cw)); cols.append(im[y0:y1, x0:x0 + cw])
            out.append(np.hstack(cols))
        Image.fromarray(np.vstack(out)).save(OUT / f"figcmp_{name}.png"); print(name, flush=True)
if __name__ == "__main__": run(int(sys.argv[1]))
