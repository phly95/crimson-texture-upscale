"""headtex_decal.py: draw clean vector shapes in SCREEN space of one mesh view and bake them into the texture through the triangles' own UVs (exact per-triangle
affine UV->screen map), so a circle drawn in the view renders as a circle on the mesh. Also mirror-copies texture content into collapsed-UV (sliver) triangles.
screen_of(model, pack, yaw, S, tris) -> per-texel screen coords for the listed triangles (first listed triangle wins where UVs overlap)"""
import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_render as HR, headtex_triid as TI
def screen_of(model, pack, yaw, S, tris, W, H, name=None, pitch=0, grow=0):
    """texel -> (x, y) screen position in the given view, for texels inside the listed triangles (UV raster, `grow` texels of padding); NaN elsewhere"""
    X, Y, Z, T = TI.proj(model, pack, yaw, pitch, S); sx = np.full((H, W), np.nan, np.float32); sy = sx.copy(); gx, gy = np.meshgrid(np.arange(W) + .5, np.arange(H) + .5)
    for j in reversed(list(tris)):                                                            # reversed: the first listed triangle is written last and wins
        vi, uv, nm = T[j]
        if name and nm != name: continue
        u = np.array(uv, np.float64) * [W, H]; A = np.c_[u, np.ones(3)]
        if abs(np.linalg.det(A)) < 1e-9: continue
        cx = np.linalg.solve(A, X[list(vi)]); cy = np.linalg.solve(A, Y[list(vi)])
        m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.round(u * 16).astype(np.int32)], 255, shift=4)
        if grow: m = cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
        k = m > 0; sx[k] = (cx[0] * gx + cx[1] * gy + cx[2])[k]; sy[k] = (cy[0] * gx + cy[1] * gy + cy[2])[k]
    return sx, sy
def bake(tex, layer_rgba, sx, sy, S):
    """composite a screen-space RGBA float layer (S x S, straight alpha 0..1, colours 0..255) into the texture through the texel->screen map"""
    k = S / layer_rgba.shape[0]; ok = np.isfinite(sx)
    mx = np.where(ok, sx / k - .5, -10).astype(np.float32); my = np.where(ok, sy / k - .5, -10).astype(np.float32)
    s = cv2.remap(layer_rgba.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = s[..., 3:4] * ok[..., None]; return np.clip(tex.astype(np.float32) * (1 - a) + s[..., :3] * a, 0, 255).astype(np.uint8)
def mirror_fill(tex, model, pack, dst, src, name=None):
    """texels of the dst (collapsed) triangles get the colour of the mirror-image surface point (x -> -x) found in the src triangles -> (tex, mask)"""
    V, T = HR.tris(model, pack); H, W = tex.shape[:2]; gx, gy = np.meshgrid(np.arange(W) + .5, np.arange(H) + .5); out = tex.copy(); mask = np.zeros((H, W), bool)
    srcs = []
    for j in src:
        vi, uv, nm = T[j]; P = V[list(vi)] * [-1, 1, 1]; srcs.append((P, np.array(uv) * [W, H]))
    for j in dst:
        vi, uv, nm = T[j]; u = np.array(uv, np.float64) * [W, H]; m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.round(u * 16).astype(np.int32)], 255, shift=4); m = cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0
        A = np.c_[u, np.ones(3)]; ys, xs = np.nonzero(m); q = np.c_[gx[m], gy[m], np.ones(m.sum())]
        try: L = np.linalg.solve(A.T, q.T).T                                                    # barycentric of each texel in the (sliver) UV triangle
        except np.linalg.LinAlgError: continue
        L = np.clip(L, -0.2, 1.2); L /= L.sum(1, keepdims=True); P3 = L @ V[list(vi)]               # 3D surface point
        best = np.full(len(P3), np.inf); su = np.zeros((len(P3), 2))
        for P, uvs in srcs:                                                                      # mirrored source triangle: barycentric of the point in its plane
            e1, e2 = P[1] - P[0], P[2] - P[0]; G = np.array([[e1 @ e1, e1 @ e2], [e1 @ e2, e2 @ e2]]); r = P3 - P[0]
            ab = np.linalg.solve(G, np.c_[r @ e1, r @ e2].T).T; bc = np.c_[1 - ab.sum(1), ab]; out_by = -np.minimum(bc.min(1), 0)
            better = out_by < best; best[better] = out_by[better]; su[better] = (bc @ uvs)[better]
        col = cv2.remap(tex, (su[:, 0] - .5).astype(np.float32)[None], (su[:, 1] - .5).astype(np.float32)[None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)[0]
        out[ys, xs] = col; mask[ys, xs] = True
    return out, mask
