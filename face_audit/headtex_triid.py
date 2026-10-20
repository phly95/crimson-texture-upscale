"""headtex_triid.py: triangle-ID render (same projection as headtex_render) -> int map (-1 = background); idsheet(model, pack, yaw, S, out) labels each visible triangle"""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_render as HR
def proj(model, pack, yaw, pitch=0, S=800):
    V, T = HR.tris(model, pack); a, b = np.radians(yaw), np.radians(pitch)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], np.float32)
    c = (V.min(0) + V.max(0)) / 2; P = (V - c) @ R.T; sc = S * 0.45 / np.abs(P[:, :2]).max()
    return P[:, 0] * sc + S / 2, -P[:, 1] * sc + S / 2, P[:, 2], T
def idmap(model, pack, yaw, pitch=0, S=800):
    X, Y, Z, T = proj(model, pack, yaw, pitch, S); ids = np.full((S, S), -1, np.int32); zb = np.full((S, S), -1e9, np.float32)
    gx, gy = np.meshgrid(np.arange(S) + .5, np.arange(S) + .5)
    for j, (vi, uv, nm) in enumerate(T):
        xs, ys, zs = X[list(vi)], Y[list(vi)], Z[list(vi)]; d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(d) < 1e-6: continue
        l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d; l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d; l2 = 1 - l0 - l1
        m = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4); z = l0 * zs[0] + l1 * zs[1] + l2 * zs[2]; m &= z > zb; ids[m] = j; zb[m] = z[m]
    return ids
def idsheet(model, pack, yaw, out, S=800, base=None):
    ids = idmap(model, pack, yaw, S=S); rng = np.random.RandomState(1); col = rng.randint(60, 255, (ids.max() + 2, 3)).astype(np.uint8)
    img = col[ids + 1] if base is None else (base * 0.6 + col[ids + 1] * 0.4).astype(np.uint8); img[ids < 0] = 30
    for j in np.unique(ids[ids >= 0]):
        ys, xs = np.nonzero(ids == j); cv2.putText(img, str(j), (int(xs.mean()) - 8, int(ys.mean()) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2); cv2.putText(img, str(j), (int(xs.mean()) - 8, int(ys.mean()) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    Image.fromarray(img).save(out)
if __name__ == "__main__":
    import headtex_figure as FG
    D = FG.ROOT / "face_audit/headtex"; t = FG.orig("fbhead1")
    for y in (180, 135, 225):
        base = HR.render(t, y, S=800, flipv=False, model=1124, pack="C1C/gamez.zip"); idsheet(1124, "C1C/gamez.zip", y, D / f"fb_ids_{y}.png", base=base)
    V, T = HR.tris(1124, "C1C/gamez.zip")
    for i, v in enumerate(V):
        m = np.where(np.linalg.norm(V - v * np.array([-1, 1, 1], np.float32), axis=1) < 1e-3)[0]; print(i, v.round(3), "mirror", m)
