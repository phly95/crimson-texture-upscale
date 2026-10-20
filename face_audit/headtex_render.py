"""headtex_render.py: software render of the cockpit pilot head mesh (planes.zip model 70) with a given texture, for checking UV layout. render(tex_rgb, yaw_deg, flipv) -> RGB uint8"""
import json, zipfile, sys
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]
_M = {}
def tris(model=70, pack="planes.zip", fixuv=False):
    if fixuv: return _fix(model, pack)
    k = (pack, model)
    if k not in _M:
        z = zipfile.ZipFile(Path.home() / "Games/CSVM/extracted" / pack); m = json.loads(z.read("models.json"))[model]
        tn = [t["name"].rsplit(".", 1)[0].lower() for t in json.loads(z.read("textures.json"))]; mats = json.loads(z.read("materials.json"))
        V = np.array([[v["x"], v["y"], v["z"]] for v in m["vertices"]], np.float32); T = []
        for p in m["polygons"]:
            mt = mats[p["materials"][0]["material_index"]].get("Textured"); name = tn[mt["texture_index"]] if mt else None
            vi = p["vertex_indices"]; uv = [(u["u"], u["v"]) for u in p["materials"][0]["uv_coords"]]
            if p["flags"]["tri_strip"]:
                for i in range(len(vi) - 2):
                    a = (vi[i], vi[i + 1], vi[i + 2]) if i % 2 == 0 else (vi[i + 1], vi[i], vi[i + 2]); u = (uv[i], uv[i + 1], uv[i + 2]) if i % 2 == 0 else (uv[i + 1], uv[i], uv[i + 2])
                    if len(set(a)) == 3: T.append((a, u, name))
            else:
                for i in range(1, len(vi) - 1): T.append(((vi[0], vi[i], vi[i + 1]), (uv[0], uv[i], uv[i + 1]), name))
        _M[k] = (V, T)
    return _M[k]
def render(tex, yaw=90, pitch=0, S=640, flipv=True, model=70, bg=(40, 40, 40), pack="planes.zip", fixuv=False, engine=False):
    """engine=True: sample like CSVM (SceneBuilder.UvAxesWithinUnitSquare): an axis whose UVs stay in [0,1] for that texture is clamped, any other axis wraps (repeat)"""
    V, T = tris(model, pack, fixuv)
    fits = {}
    if engine:
        for vi, uv, nm in T:
            f = fits.setdefault(nm, [True, True])
            for q in uv: f[0] &= -1e-4 <= q[0] <= 1 + 1e-4; f[1] &= -1e-4 <= q[1] <= 1 + 1e-4
    a, b = np.radians(yaw), np.radians(pitch)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], np.float32)
    c = (V.min(0) + V.max(0)) / 2; P = (V - c) @ R.T; sc = S * 0.45 / np.abs(P[:, :2]).max()
    X = P[:, 0] * sc + S / 2; Y = -P[:, 1] * sc + S / 2; Z = P[:, 2]
    img = np.zeros((S, S, 3), np.uint8); img[:] = bg; zb = np.full((S, S), -1e9, np.float32)
    for vi, uv, nm in T:
        tx = tex[nm] if isinstance(tex, dict) else tex
        if tx is None: continue
        th, tw = tx.shape[:2]
        xs, ys, zs = X[list(vi)], Y[list(vi)], Z[list(vi)]
        x0, x1, y0, y1 = int(max(0, xs.min())), int(min(S - 1, xs.max() + 1)), int(max(0, ys.min())), int(min(S - 1, ys.max() + 1))
        if x1 <= x0 or y1 <= y0: continue
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(d) < 1e-6: continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + .5, np.arange(y0, y1 + 1) + .5)
        l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d; l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d; l2 = 1 - l0 - l1
        m = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
        z = l0 * zs[0] + l1 * zs[1] + l2 * zs[2]; sl = zb[y0:y1 + 1, x0:x1 + 1]; m &= z > sl
        if not m.any(): continue
        u = l0 * uv[0][0] + l1 * uv[1][0] + l2 * uv[2][0]; v = l0 * uv[0][1] + l1 * uv[1][1] + l2 * uv[2][1]
        if flipv: v = 1 - v
        mx, my = (u * tw - .5).astype(np.float32), (v * th - .5).astype(np.float32)
        if engine:
            fu, fv = fits[nm]; mx = np.clip(mx, 0, tw - 1) if fu else np.mod(mx, tw).astype(np.float32); my = np.clip(my, 0, th - 1) if fv else np.mod(my, th).astype(np.float32)
            col = cv2.remap(tx, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        else: col = cv2.remap(tx, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        sub = img[y0:y1 + 1, x0:x1 + 1]; sub[m] = col[m]; sl[m] = z[m]
    return img
if __name__ == "__main__":
    from PIL import Image
    t = np.array(Image.open(sys.argv[1]).convert("RGB")); out = sys.argv[2]
    row = lambda fv: np.hstack([render(t, y, flipv=fv, S=400) for y in (0, 45, 90, 135, 180)])
    Image.fromarray(np.vstack([row(True), row(False)])).save(out)


def density(V, T):
    """texels^2 per unit 3D area of every triangle, relative to the median (1024 px UV space); ~0 = the UVs of that triangle are collapsed onto a line"""
    d = []
    for vi, uv, nm in T:
        P = V[list(vi)]; a3 = np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0])) / 2; u = np.array(uv) * 1024
        d.append(abs((u[1, 0] - u[0, 0]) * (u[2, 1] - u[0, 1]) - (u[1, 1] - u[0, 1]) * (u[2, 0] - u[0, 0])) / 2 / max(a3, 1e-12))
    d = np.array(d); return d / np.median(d)
def broken(V, T, thr=0.05): return [i for i, r in enumerate(density(V, T)) if r < thr]
def _fix(model, pack):
    """reference-only: triangles whose UVs are collapsed (the original asset's right goggle) borrow the UVs of their mirror-image triangle, so a render of the mesh shows both sides alike"""
    V, T = tris(model, pack); bad = set(broken(V, T)); mir = {}
    for i, v in enumerate(V):
        m = np.where(np.linalg.norm(V - v * np.array([-1, 1, 1], np.float32), axis=1) < 1e-4)[0]
        if len(m): mir[i] = int(m[0])
    key = {}
    for j, (vi, uv, nm) in enumerate(T):
        if j not in bad: key[frozenset(vi)] = (vi, uv)
    out = []
    for j, (vi, uv, nm) in enumerate(T):
        if j in bad and all(x in mir for x in vi) and frozenset(mir[x] for x in vi) in key:
            mvi, muv = key[frozenset(mir[x] for x in vi)]; uvmap = {a: b for a, b in zip(mvi, muv)}; uv = tuple(uvmap[mir[x]] for x in vi)
        out.append((vi, uv, nm))
    return V, out
