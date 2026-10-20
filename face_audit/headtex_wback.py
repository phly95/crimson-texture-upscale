"""headtex_wback.py [SRC] [OUT]: wrap the woman's leather helmet around the BACK of her head. The back-of-head triangles of womanhead (model 1175 of C1/gamez.zip) have collapsed or sliver UVs: 5,9 (a line (0.062,0.699)->(0.293,0.93)), 10,11 (a line down the left edge), 12,24 (thin wedge)
and 3,4 (thin diagonal wedge) sample the plain atlas margin, and 22,23,27,28 stretch a band that contains the helmet outline into two dark arcs. Those texels are repainted as smooth dark helmet leather (a colour taken from the dome,
soft top-to-bottom shading, a fine seam stitch line) so the collapsed triangles read as a continuous helmet back -> headtex/womanhead_back_2048.png"""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_render as HR
D = Path(__file__).resolve().parent / "headtex"
BACK = [3, 4, 5, 9, 10, 11, 12, 24, 22, 23, 27, 28]
def paint(src, grow=14, seam=True):
    tex = np.array(Image.open(src).convert("RGB")); H, W = tex.shape[:2]
    V, T = HR.tris(1175, "C1/gamez.zip"); M = np.zeros((H, W), np.uint8)
    for j in BACK:
        p = np.array(T[j][1]) * [W, H]; p[:, 1] = np.clip(p[:, 1], 0, H - 1); cv2.fillPoly(M, [np.round(p).astype(np.int32)], 255, cv2.LINE_AA)
        for a, b in ((0, 1), (1, 2), (2, 0)): cv2.line(M, tuple(np.round(p[a]).astype(int)), tuple(np.round(p[b]).astype(int)), 255, 3)               # the collapsed ones are lines: give them width
    M = cv2.dilate(M, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    dome = np.median(tex[int(0.30 * H):int(0.42 * H), int(0.35 * W):int(0.55 * W)].reshape(-1, 3), axis=0)                                # leather of the helmet's rear side (no crown highlight)
    ys = np.linspace(0, 1, H)[:, None, None]; lum = 1.10 - 0.32 * ys                                                                       # soft shading: a little lighter at the crown, darker toward the nape
    fill = np.broadcast_to(np.clip(dome[None, None, :] * lum, 0, 255), (H, W, 3)).copy()
    if seam:                                                                                                                                # rear strap band across the back: dark groove edges, a lighter leather strip, stitch dashes
        y0, y1 = int(0.50 * H), int(0.575 * H); xe = int(0.12 * W)
        fill[y0:y1, :xe] = np.clip(dome * 1.18, 0, 255); cv2.line(fill, (0, y0), (xe, y0), (20, 14, 12), 6, cv2.LINE_AA); cv2.line(fill, (0, y1), (xe, y1), (20, 14, 12), 6, cv2.LINE_AA)
        for yy in (y0 + 14, y1 - 14):
            for x in range(0, xe, 22): cv2.line(fill, (x, yy), (x + 11, yy), (150, 122, 100), 3, cv2.LINE_AA)
    fill = cv2.GaussianBlur(fill, (0, 0), 1.6)
    a = cv2.GaussianBlur(M.astype(np.float32) / 255, (0, 0), 6)[..., None]
    return np.clip(fill * a + tex * (1 - a) + .5, 0, 255).astype(np.uint8)
if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else str(D / "womanhead_flap_2048.png"); out = sys.argv[2] if len(sys.argv) > 2 else str(D / "womanhead_back_2048.png")
    Image.fromarray(paint(src)).save(out); print(out)
