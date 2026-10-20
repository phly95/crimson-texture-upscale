"""headtex_gloves.py: hand-drawn glove/boot leather tile (512x512 from a 2x supersampled drawing) -> headtex/gloves_drawn.png.
The hand is a 9-triangle block and the boot a 6-triangle wedge that sample the WHOLE sheet per face (u 0..1, v 0..2 = the sheet repeats twice along the length), so the tile is drawn to read as a glove when repeated:
four finger panels (rounded cel-shaded leather) with dark seams and tan stitch dashes, two knuckle creases per panel, and a stitched hem band on the v edge (cuff / boot top / sole line where the tile repeats)."""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_gen as G
D = G.ROOT / "face_audit/headtex"
LEATHER = np.array([34, 24, 19]); DARK = np.array([13, 9, 8]); HI = np.array([62, 44, 33]); HI2 = np.array([104, 76, 56]); STITCH = (170, 136, 96); SEAM = (7, 5, 4)
def make(S=512, SS=2):
    W = S * SS; img = np.zeros((W, W, 3), np.float32); img[:] = DARK
    n = 4; cw = W / n; hem = int(0.085 * W)
    for i in range(n):
        x0, x1 = i * cw, (i + 1) * cw; xs = np.arange(W, dtype=np.float32)[None, :]; ys = np.arange(W, dtype=np.float32)[:, None]
        t = np.clip((xs - x0) / cw, 0, 1)                                                          # 0..1 across the panel
        bulge = np.sin(np.pi * t) ** 0.8                                                           # rounded finger: light in the middle, dark at the seams
        shade = bulge * (0.55 + 0.45 * np.clip(1 - np.abs(ys - W * 0.5) / (W * 0.5), 0, 1))
        band = np.digitize(shade, [0.30, 0.64, 0.90]).astype(np.float32)                           # 4 flat cel bands
        cols = np.stack([DARK, LEATHER, HI, HI2]); m = (xs >= x0) & (xs < x1) & (ys >= hem)
        for b in range(4):
            mm = m & (band == b); img[np.broadcast_to(mm, (W, W))] = cols[b]
    ov = np.clip(img, 0, 255).astype(np.uint8).copy()
    for i in range(n):
        cx = int((i + 0.5) * cw)                                                                   # knuckle creases: two arcs per panel, dark with a light lip below
        for v, bow in ((0.34, 0.035), (0.67, 0.035)):
            cy = int(v * W); pts = np.array([[int(cx + (u - 0.5) * cw * 0.78), int(cy + bow * W * (1 - (2 * (u - 0.5)) ** 2) * -1)] for u in np.linspace(0, 1, 40)], np.int32)
            cv2.polylines(ov, [pts], False, SEAM, max(3, SS * 3), cv2.LINE_AA); cv2.polylines(ov, [pts + [0, SS * 4]], False, tuple(int(c) for c in (96, 70, 52)), max(2, SS * 2), cv2.LINE_AA)
    for i in range(1, n):                                                                          # finger seams with stitch dashes either side
        x = int(i * cw); cv2.line(ov, (x, hem), (x, W), SEAM, SS * 5, cv2.LINE_AA)
        for off in (-SS * 9, SS * 9):
            for y in range(hem + SS * 8, W, SS * 16): cv2.line(ov, (x + off, y), (x + off, y + SS * 9), STITCH, SS * 2, cv2.LINE_AA)
    cv2.rectangle(ov, (0, 0), (W, hem), tuple(int(c) for c in (22, 16, 14)), -1)                    # hem band (cuff / boot top / sole line where the tile repeats)
    cv2.line(ov, (0, hem), (W, hem), SEAM, SS * 5, cv2.LINE_AA)
    for yy in (hem // 2,):
        for x in range(SS * 6, W, SS * 16): cv2.line(ov, (x, yy), (x + SS * 9, yy), STITCH, SS * 2, cv2.LINE_AA)
    out = cv2.resize(ov, (S, S), interpolation=cv2.INTER_AREA); Image.fromarray(out).save(D / "gloves_drawn.png"); return out
if __name__ == "__main__": make()
