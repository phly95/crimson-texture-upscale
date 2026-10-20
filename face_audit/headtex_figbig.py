"""headtex_figbig.py [S]: high-resolution full-figure sheets of every character (original on top, hand-drawn below; front, 3/4, side, back; corrected mesh) cropped to the figure -> gallery/site/headtex/wip/figbig_<name>.png"""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_figure as FG, headtex_render as HR
D = FG.ROOT / "face_audit/headtex"; W = FG.ROOT / "gallery/site/headtex/wip"; BG = (40, 40, 40)
L = lambda f: np.array(Image.open(D / f).convert("RGB"))
def sets():
    gl = L("gloves_drawn.png")
    return {"pilot": ("pilot", {"cphead": L("cphead_drawn.png"), "cpsuit128": L("cpsuit128_drawn.png")}),
            "copilot": ("copilot", {"cphead02": L("cphead02_drawn.png"), "cpsuit128_01": L("cpsuit128_01_drawn.png"), "cpsuit128_02": L("cpsuit128_02_drawn.png"), "gloves": gl}),
            "woman": ("woman", {"womanhead": L("womanhead_flap_2048.png"), "womanface": L("womanface_c_256_x2.png"), "womanarm": L("womanarm_c_512_x2.png"), "womanchest": L("womanchest_cartoon_512x1024.png"), "woman1": L("woman1_cartoon_512x1024.png"), "gloves": gl}),
            "oldman": ("fbman", {"fbhead1": L("fbhead1_cartoon_2048_v2.png"), "fbsuit128_01": L("fbsuit128_01_final.png"), "fbsuit128_02": L("fbsuit128_02_final.png"), "gloves": gl})}
def run(S=1500, yaws=(180, 135, 90, 0)):
    for name, (fg, tex) in sets().items():
        key, pack = FG.fig(fg, True); rows = []
        for t in (FG.texdict(), FG.texdict(tex)):
            ims = [HR.render(t, y, 0, S=S, flipv=False, model=key, pack=pack, engine=True, bg=BG) for y in yaws]; rows.append(ims)
        m = np.zeros((S, S), bool)
        for r in rows:
            for im in r: m |= (np.abs(im.astype(int) - BG).max(-1) > 6)
        ys, xs = np.nonzero(m); y0, y1 = max(ys.min() - 12, 0), min(ys.max() + 12, S); cw = (xs.max() - xs.min()) + 24
        out = []
        for r in rows:
            cols = []
            for im in r:
                cx = int(np.nonzero((np.abs(im.astype(int) - BG).max(-1) > 6).any(0))[0].mean()) if (np.abs(im.astype(int) - BG).max(-1) > 6).any() else S // 2
                x0 = int(np.clip(cx - cw // 2, 0, S - cw)); cols.append(im[y0:y1, x0:x0 + cw])
            out.append(np.hstack(cols))
        sheet = np.vstack(out); Image.fromarray(sheet).save(W / f"figbig_{name}.png"); print(name, sheet.shape, flush=True)
if __name__ == "__main__": run(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
