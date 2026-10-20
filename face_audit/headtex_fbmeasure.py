"""headtex_fbmeasure.py TEX: numbers for the fbhead1 cartoon texture: (1) roundness of the visible glasses lens in the front render (circle fit to the dark rim, radial RMS / radius,
fitted-ellipse axis ratio), (2) grey / background-glow texels inside the UV area the mesh uses (engine wrap), (3) low-frequency correlation to the original over the used area"""
import sys, json
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_render as HR, headtex_figure as FG, headtex_fbcompose as C, headtex_fin as F
MODEL, PACK = 1124, "C1C/gamez.zip"
def used_mask(N):
    V, T = HR.tris(MODEL, PACK); m = np.zeros((2 * N, 2 * N), np.uint8)
    for vi, uv, nm in T: cv2.fillPoly(m, [np.round(np.array(uv) * N * 16).astype(np.int32)], 255, shift=4)
    w = m[:N, :N] | m[N:, :N] | m[:N, N:] | m[N:, N:]; return w > 0                                   # v (and u) beyond 1 wrap
def lens(tex, S=1600):
    prm = json.load(open(C.D / "fbhead1_c2_params.json")); g = prm["gl"]; k = S / C.S
    im = HR.render(FG.texdict({"fbhead1": tex}), 180, S=S, flipv=False, model=MODEL, pack=PACK, engine=True)
    cx, cy, r = (g["cx"] + C.OX) * k, (g["cy"] + C.OY) * k, g["r"] * k
    lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(int); dark = lab[..., 0] < 75; yy, xx = np.mgrid[0:S, 0:S]; d = np.hypot(xx - cx, yy - cy)
    sel = dark & (np.abs(d - r) < g["w"] * k * 1.0); ang = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360
    rad = []                                                                                         # per 2-degree sector: centre of the dark rim run (robust to lashes/brows touching the band)
    for a0 in range(0, 360, 2):
        m = sel & (ang >= a0) & (ang < a0 + 2)
        if m.sum() >= 3: rad.append((a0 + 1, float(np.median(d[m]))))
    A_, R_ = np.radians(np.array(rad)[:, 0]), np.array(rad)[:, 1]; x, y = cx + R_ * np.cos(A_), cy + R_ * np.sin(A_)
    M = np.c_[2 * x, 2 * y, np.ones(len(x))]; s_ = np.linalg.lstsq(M, x * x + y * y, rcond=None)[0]; c = s_[:2]; R = np.sqrt(s_[2] + c @ c); res = np.hypot(x - c[0], y - c[1]) - R
    e = cv2.fitEllipse(np.c_[x, y].astype(np.float32))
    return dict(radius_px=round(float(R), 1), radial_rms_px=round(float(np.sqrt((res ** 2).mean())), 2), max_dev_px=round(float(np.abs(res).max()), 2), rms_over_r=round(float(np.sqrt((res ** 2).mean()) / R), 4),
                ellipse_axis_ratio=round(min(e[1]) / max(e[1]), 4), sectors=len(rad), render_S=S)
def grey(tex):
    N = tex.shape[0]; u = used_mask(N); lab = cv2.cvtColor(tex, cv2.COLOR_RGB2LAB).astype(int); L, a, b = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
    g = (np.hypot(a, b) < 4) & (L > 20) & (L < 60); cyan = (b < -4) & (L > 40)
    return dict(grey_frac=round(float((g & u).sum() / u.sum()), 5), cyan_frac=round(float((cyan & u).sum() / u.sum()), 5), used_frac=round(float(u.mean()), 3))
def lowcorr(tex):
    o = FG.orig("fbhead1"); N = tex.shape[0]; o = cv2.resize(o, (N, N), interpolation=cv2.INTER_CUBIC); u = used_mask(N)
    g = lambda x: cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (0, 0), 6 * N / 1024).astype(np.float32)
    a, b = g(tex)[u], g(o)[u]; a -= a.mean(); b -= b.mean(); return round(float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())), 3)
if __name__ == "__main__":
    t = np.array(Image.open(sys.argv[1]).convert("RGB")); print(json.dumps(dict(lens=lens(t), **grey(t), lowcorr_to_orig=lowcorr(t)), indent=1))
