"""headtex_bake.py: render the pilot head mesh from several views, and bake (redrawn) view images back into the mesh's UV layout.
render_view(tex, yaw, pitch) -> (rgb, depth, ...) ; bake(views) -> (tex_rgb, weight)"""
import sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_render as HR
S = 1024
def cam(V, yaw, pitch):
    a, b = np.radians(yaw), np.radians(pitch)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], np.float32)
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]], np.float32)
    R = Rx @ Ry; c = (V.min(0) + V.max(0)) / 2; sc = S * 0.46 / np.linalg.norm(V - c, axis=1).max(); return R, c, sc
def project(V, yaw, pitch):
    R, c, sc = cam(V, yaw, pitch); P = (V - c) @ R.T; return P[:, 0] * sc + S / 2, -P[:, 1] * sc + S / 2, P[:, 2]
def render_view(tex, yaw, pitch=0, model=70, bg=(128, 128, 128), pack="planes.zip", fixuv=True):
    V, T = HR.tris(model, pack, fixuv); X, Y, Z = project(V, yaw, pitch)
    img = np.zeros((S, S, 3), np.uint8); img[:] = bg; zb = np.full((S, S), -1e9, np.float32)
    for vi, uv, nm in T:
        tx = tex[nm] if isinstance(tex, dict) else tex
        if tx is None: continue
        th, tw = tx.shape[:2]
        xs, ys, zs = X[list(vi)], Y[list(vi)], Z[list(vi)]
        x0, x1, y0, y1 = int(max(0, xs.min())), int(min(S - 1, xs.max() + 1)), int(max(0, ys.min())), int(min(S - 1, ys.max() + 1))
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if x1 <= x0 or y1 <= y0 or abs(d) < 1e-6: continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + .5, np.arange(y0, y1 + 1) + .5)
        l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d; l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d; l2 = 1 - l0 - l1
        m = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4); z = l0 * zs[0] + l1 * zs[1] + l2 * zs[2]; sl = zb[y0:y1 + 1, x0:x1 + 1]; m &= z > sl
        if not m.any(): continue
        u = l0 * uv[0][0] + l1 * uv[1][0] + l2 * uv[2][0]; v = l0 * uv[0][1] + l1 * uv[1][1] + l2 * uv[2][1]
        col = cv2.remap(tx, (u * tw - .5).astype(np.float32), (v * th - .5).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        sub = img[y0:y1 + 1, x0:x1 + 1]; sub[m] = col[m]; sl[m] = z[m]
    return img, zb
def bake(views, model=70, TS=1024, pw=3.0, eps=0.004, erode=45, pack="planes.zip", only=None, TSH=None, skip_extra=()):
    """views: list of (yaw, pitch, rgb image SxSx3, depth buffer from render_view). Returns baked (TS,TS,3) float and weight (TS,TS)."""
    V, T = HR.tris(model, pack); c = (V.min(0) + V.max(0)) / 2
    acc = np.zeros((TS, TS, 3), np.float64); wsum = np.zeros((TS, TS), np.float64)
    er = lambda zb: cv2.erode((zb > -1e8).astype(np.uint8), np.ones((erode, erode), np.uint8)) > 0       # keep clear of the redraw's halo at the silhouette
    prj = [(project(V, y, p), cam(V, y, p)[0], im.astype(np.float32), zb, er(zb)) for y, p, im, zb in views]
    TSW, TS = (TS, TSH or TS)
    acc = np.zeros((TS, TSW, 3), np.float64); wsum = np.zeros((TS, TSW), np.float64)
    skip = set(HR.broken(V, T)) | set(skip_extra)                                              # collapsed UVs (original's right goggle): nothing sensible can be baked there
    for j, (vi, uv, nm) in enumerate(T):
        if only and nm != only: continue
        if j in skip: continue
        P3 = V[list(vi)]; n = np.cross(P3[1] - P3[0], P3[2] - P3[0]); ln = np.linalg.norm(n)
        if ln < 1e-12: continue
        n /= ln
        if np.dot(n, P3.mean(0) - c) < 0: n = -n
        xs = np.array([u[0] for u in uv]) * TSW; ys = np.array([u[1] for u in uv]) * TS
        x0, x1, y0, y1 = int(max(0, np.floor(xs.min()) - 1)), int(min(TSW - 1, np.ceil(xs.max()) + 1)), int(max(0, np.floor(ys.min()) - 1)), int(min(TS - 1, np.ceil(ys.max()) + 1))
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if x1 <= x0 or y1 <= y0 or abs(d) < 1e-9: continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + .5, np.arange(y0, y1 + 1) + .5)
        l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d; l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d; l2 = 1 - l0 - l1
        m = (l0 >= -0.02) & (l1 >= -0.02) & (l2 >= -0.02)                    # a little outside the triangle = built-in padding
        l0, l1, l2 = [np.clip(a, 0, 1) for a in (l0, l1, l2)]; s = l0 + l1 + l2; l0, l1, l2 = l0 / s, l1 / s, l2 / s
        W3 = l0[..., None] * P3[0] + l1[..., None] * P3[1] + l2[..., None] * P3[2]       # 3D position per texel
        tw_ = np.zeros(m.shape); tc = np.zeros(m.shape + (3,))
        for (X, Y, Z), R, im, zb, em in prj:
            Pc = (W3 - c) @ R.T; sc = S * 0.46 / np.linalg.norm(V - c, axis=1).max()
            px = Pc[..., 0] * sc + S / 2; py = -Pc[..., 1] * sc + S / 2; pz = Pc[..., 2]
            cosv = (n @ R.T)[2]                                                  # normal's z in view space; >0 faces the camera
            if cosv <= 0.05: continue
            ix = np.clip(px.astype(int), 0, S - 1); iy = np.clip(py.astype(int), 0, S - 1)
            vis = pz >= zb[iy, ix] - eps * (V.max(0) - V.min(0)).max() * sc * 3
            w = (cosv ** pw) * vis * em[iy, ix]
            col = cv2.remap(im, px.astype(np.float32) - .5, py.astype(np.float32) - .5, cv2.INTER_LINEAR)
            tw_ += w; tc += w[..., None] * col
        sl = (slice(y0, y1 + 1), slice(x0, x1 + 1)); mm = m & (tw_ > 0)
        acc[sl][mm] += tc[mm]; wsum[sl][mm] += tw_[mm]
    with np.errstate(invalid="ignore", divide="ignore"): out = acc / wsum[..., None]
    return np.nan_to_num(out), wsum
