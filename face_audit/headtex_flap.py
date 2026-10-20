"""headtex_flap.py IN OUT: replace the klein-drawn ear flap of the woman's helmet by a vector-drawn perfect circle (in screen space of the profile view: local UV->screen transform inverted), in cartoon style"""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_render as HR
CENTER = np.array([0.455, 0.575]); RS = 92.0                                                    # flap centre (UV) and radius in screen px at S=800 (profile view)
def jacobian(model=1175, pack="C1/gamez.zip", S=800, yaw=90):
    V, T = HR.tris(model, pack); a = np.radians(yaw); R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], np.float32)
    c = (V.min(0) + V.max(0)) / 2; P = (V - c) @ R.T; sc = S * 0.45 / np.abs(P[:, :2]).max(); X = P[:, 0] * sc + S / 2; Y = -P[:, 1] * sc + S / 2
    for vi, uv, nm in T:
        if nm != "womanhead": continue
        (x0, y0), (x1, y1), (x2, y2) = uv; d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(d) < 1e-9: continue
        l0 = ((y1 - y2) * (CENTER[0] - x2) + (x2 - x1) * (CENTER[1] - y2)) / d; l1 = ((y2 - y0) * (CENTER[0] - x2) + (x0 - x2) * (CENTER[1] - y2)) / d
        if min(l0, l1, 1 - l0 - l1) >= 0:
            A = np.array([[u[0], u[1], 1] for u in uv]); sx = np.linalg.solve(A, X[list(vi)]); sy = np.linalg.solve(A, Y[list(vi)]); return np.array([[sx[0], sx[1]], [sy[0], sy[1]]])
def draw(tex):
    H, W = tex.shape[:2]; J = jacobian(); Ji = np.linalg.inv(J); K = 2; Hs, Ws = H * K, W * K
    def poly(rs, n=360):
        th = np.linspace(0, 2 * np.pi, n, endpoint=False); pts = (Ji @ (rs * np.stack([np.cos(th), np.sin(th)]))).T + CENTER          # UV
        return np.round(pts * [Ws, Hs]).astype(np.int32)
    img = cv2.resize(tex, (Ws, Hs), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    # 1) erase the old flap: disc a bit larger than it, filled with the helmet colour sampled just outside, soft edge
    ring = np.zeros((Hs, Ws), np.uint8); cv2.fillPoly(ring, [poly(RS * 1.28)], 255); cv2.fillPoly(ring, [poly(RS * 1.10)], 0); helm = img[ring > 0].mean(0)
    er = np.zeros((Hs, Ws), np.uint8); cv2.fillPoly(er, [poly(RS * 1.12)], 255); er = cv2.GaussianBlur(er.astype(np.float32) / 255, (0, 0), 6 * K)[..., None]
    yy, xx = np.mgrid[0:Hs, 0:Ws].astype(np.float32); cx, cy = CENTER[0] * Ws, CENTER[1] * Hs; g = 1.0 + 0.10 * ((cx - xx) + (cy - yy)) / (0.3 * Ws)       # gentle light from the upper left
    img = img * (1 - er) + (helm * np.clip(g, 0.8, 1.2)[..., None]) * er
    # 2) vector flap
    def fill(rs, colour, soft=1.5):
        m = np.zeros((Hs, Ws), np.uint8); cv2.fillPoly(m, [poly(rs)], 255, cv2.LINE_AA); a = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), soft * K)[..., None]; return a
    def paint(a, col): nonlocal img; img = img * (1 - a) + np.array(col, np.float32) * a
    paint(fill(RS, 0), (20, 14, 12))                                                                                                # outline
    body = fill(RS - 6, 0); lit = np.clip(1.0 + 0.55 * ((cx - xx) + (cy - yy)) / (0.28 * Ws), 0.55, 1.45)[..., None]                    # body leather with a diagonal gradient
    paint(body, (0, 0, 0)); img = img * (1 - 0) ; base = np.array([56, 41, 36], np.float32) * lit; img = img * (1 - body) + base * body
    paint(fill(RS * 0.80 + 3, 0), (24, 17, 15)); inner = fill(RS * 0.80 - 2, 0); ib = np.array([62, 46, 40], np.float32) * lit; img = img * (1 - inner) + ib * inner  # raised inner ring
    # stitching between the rim and the inner ring
    for t in np.linspace(0, 2 * np.pi, 40, endpoint=False):
        p0 = (Ji @ (RS * 0.90 * np.array([np.cos(t), np.sin(t)]))) + CENTER; p1 = (Ji @ (RS * 0.90 * np.array([np.cos(t + 0.09), np.sin(t + 0.09)]))) + CENTER
        m = np.zeros((Hs, Ws), np.uint8); cv2.line(m, tuple(np.round(p0 * [Ws, Hs]).astype(int)), tuple(np.round(p1 * [Ws, Hs]).astype(int)), 255, int(2.2 * K), cv2.LINE_AA); a = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 0.8 * K)[..., None]; paint(a * 0.8, (118, 94, 80))
    # soft highlight crescent, upper left inside the inner ring
    hl = fill(RS * 0.62, 0); sh = np.zeros((Hs, Ws), np.uint8); cv2.fillPoly(sh, [poly(RS * 0.62) + [int(-0.05 * Ws * 0.05), int(-0.05 * Hs * 0.05)]], 255, cv2.LINE_AA)
    cres = np.clip(hl - fill(RS * 0.60, 0)[:, :, :] * 0, 0, 1) * 0
    glow = np.clip(1 - np.hypot(xx - (cx - 0.035 * Ws), yy - (cy - 0.035 * Hs)) / (0.07 * Ws), 0, 1)[..., None] * inner * 0.28; paint(glow, (150, 122, 108))
    return cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA)
if __name__ == "__main__": Image.fromarray(draw(np.array(Image.open(sys.argv[1]).convert("RGB")))).save(sys.argv[2])
