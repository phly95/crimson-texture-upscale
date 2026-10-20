"""headtex_trace.py: vector traces of the features of a REFERENCE head render (goggle lenses+frames, skin, helmet leather, chin strap) as simplified polylines, drawn over the render for checking"""
import sys, json
from pathlib import Path
import cv2, numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "face_audit/headtex"
def classes(rgb, mask):
    """label map: 0 outside mesh, 1 skin, 2 leather/hair (dark brown), 3 goggle (dark grey/teal, low chroma) -- simple Lab rules tuned on the front render"""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32); L, a, b = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
    lab_ = np.zeros(mask.shape, np.uint8)
    skin = (L > 45) & (a > 4) & (b > 8); goggle = (L < 40) & ((a + b) < 10) & (np.hypot(a, b) < 14); leather = ~skin & ~goggle
    lab_[mask & leather] = 2; lab_[mask & skin] = 1; lab_[mask & goggle] = 3
    return lab_
def polys(lbl, k, eps=2.0, min_area=300):
    m = cv2.morphologyEx((lbl == k).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE); out = []
    for c in cs:
        if cv2.contourArea(c) >= min_area: out.append(cv2.approxPolyDP(c, eps, True)[:, 0, :])
    return out
if __name__ == "__main__":
    tg, k = sys.argv[1], sys.argv[2]
    rgb = np.array(Image.open(D / f"v_{tg}_{k}.png").convert("RGB")); mask = np.load(D / f"v_{tg}_{k}_z.npy") > -1e8
    lbl = classes(rgb, mask); cols = {1: (255, 200, 0), 2: (0, 200, 255), 3: (255, 0, 255)}; ov = rgb.copy()
    tint = np.zeros_like(rgb)
    for kk, c in cols.items(): tint[lbl == kk] = c
    ov = (rgb * .55 + tint * .45).astype(np.uint8)
    for kk, c in cols.items():
        for p in polys(lbl, kk): cv2.polylines(ov, [p.reshape(-1, 1, 2)], True, (255, 255, 255), 2)
    x0, x1, y0, y1 = [int(v) for v in np.r_[np.where(mask.any(0))[0][[0, -1]], np.where(mask.any(1))[0][[0, -1]]]]
    Image.fromarray(np.hstack([rgb, ov])[y0:y1, :]).save(D / f"trace_{tg}_{k}.png"); np.save(D / f"lbl_{tg}_{k}.npy", lbl); print("ok", x0, x1, y0, y1)
