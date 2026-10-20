"""zach_tube.py [fit|render]: the fuselage as a cylinder (orthographic camera). The 48gypsy decal is wrapped on it: texture u,v -> (axial offset, arc length) -> 3D point -> image.
Parameters (a, b, R, th0, phi, sig, cx, cy) are fitted to landmark pairs (6 text/card points) and to the traced oval edge of the original (6b4b3dd2 OG, 6x crop grid)."""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit")
import numpy as np, cv2
from scipy.optimize import least_squares
from scipy.interpolate import LinearNDInterpolator
Z = "/home/deck/workspace/csvm-hd/face_audit/zach/"
LM = [((75, 310), (100, 400)), ((385, 565), (470, 440)), ((130, 545), (210, 545)), ((415, 690), (545, 510)), ((300, 430), (345, 345)), ((300, 705), (330, 625))]
EDGE = np.array([(88, 68), (200, 125), (300, 215), (380, 300), (440, 380), (470, 470), (465, 540), (62, 170), (60, 250), (300, 710), (220, 690), (150, 640), (380, 690)], float)
TC = np.array([325.0, 255.0]); TR = 250.0
def fwd(p, uv):
    a, b, R, th0, phi, sig, cx, cy = p
    ax = np.array([np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), np.sin(b)])
    z = np.array([0, 0, 1.0]); e1 = z - (z @ ax) * ax; e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
    d = uv - TC; pp = d[:, 0] * np.cos(phi) - d[:, 1] * np.sin(phi); qq = d[:, 0] * np.sin(phi) + d[:, 1] * np.cos(phi)
    t = sig * pp; th = th0 + sig * qq / R
    P = t[:, None] * ax + R * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
    P0 = R * (np.cos(th0) * e1 + np.sin(th0) * e2)
    return np.c_[cx + (P - P0)[:, 0], cy + (P - P0)[:, 1]], P[:, 2] - P0[2]
from PIL import Image
_al = (np.asarray(Image.open(Z + "gypsy_hd.png").convert("RGBA").resize((640, 640)))[..., 3] > 40).astype(np.uint8)
_cs, _ = cv2.findContours(_al, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); _c = max(_cs, key=len)[:, 0, :].astype(float)
CIRC = _c[::max(1, len(_c) // 140)]
src_pts = np.array([a for a, b in LM], float); tex_pts = np.array([b for a, b in LM], float)
def resid(p, wedge=0.6):
    m, _ = fwd(p, tex_pts); r1 = (m - src_pts).ravel() / 20.0
    c, _ = fwd(p, CIRC); dist = np.linalg.norm(EDGE[:, None, :] - c[None, :, :], axis=2).min(1)
    r2 = wedge * dist / 20.0
    return np.r_[r1, r2, 0.3 * (p[0] - np.deg2rad(31)) / 0.5]
if sys.argv[1] == "fit":
    rng = np.random.default_rng(1); best = None
    for k in range(300):
        p0 = [np.deg2rad(31 + rng.normal(0, 12)), rng.uniform(-0.6, 0.6), rng.uniform(150, 2500), rng.uniform(-1.0, 1.0), rng.normal(0, 0.4), rng.uniform(0.4, 1.4), 270, 400]
        lo = [-1.2, -1.2, 120, -1.5, -1.5, 0.2, 0, 0]; hi = [1.8, 1.2, 4000, 1.5, 1.5, 2.5, 720, 840]
        try:
            r = least_squares(resid, p0, bounds=(lo, hi))
        except Exception:
            continue
        if best is None or r.cost < best.cost: best = r
    p = best.x; print("cost", round(best.cost, 2), "params a=%.1fdeg b=%.1fdeg R=%.0f th0=%.1fdeg phi=%.1fdeg sig=%.2f c=(%.0f,%.0f)" % (np.rad2deg(p[0]), np.rad2deg(p[1]), p[2], np.rad2deg(p[3]), np.rad2deg(p[4]), p[5], p[6], p[7]))
    m, _ = fwd(p, tex_pts); print("landmark err px", np.round(np.linalg.norm(m - src_pts, axis=1), 1))
    json.dump(list(p), open(Z + "tube.json", "w"))
def inv_interp(p, step=6):
    uu, vv = np.meshgrid(np.arange(-40, 700, step), np.arange(-40, 700, step)); uv = np.c_[uu.ravel(), vv.ravel()].astype(float)
    xy, depth = fwd(p, uv)
    th_vis = depth > -1e9
    return LinearNDInterpolator(xy, uv)
if sys.argv[1] == "check":
    from PIL import Image
    import os
    p = json.load(open(Z + "tube.json")); I = inv_interp(p)
    gy, gx = np.mgrid[0:840, 0:720]; T = I(gx, gy).astype(np.float32)          # (840,720,2) texture view coords, NaN outside
    tex = np.asarray(Image.open(Z + "gypsy_hd.png").convert("RGBA"), np.float32) / 255
    mx, my = T[..., 0] * 1.6, T[..., 1] * 1.6
    nan = np.isnan(mx); mx[nan] = -10; my[nan] = -10
    wp = cv2.remap(tex, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    wp[..., :3] = wp[..., :3].mean(-1, keepdims=True); a = wp[..., 3:]
    o6 = np.asarray(Image.open(Z + "wrap_og.png"), np.float32) / 255
    ov = o6 * (1 - 0.6 * a) + wp[..., :3] * 0.6 * a
    Image.fromarray((np.concatenate([o6, ov], 1) * 255).astype(np.uint8)).save(Z + f"tube_check{os.environ.get('TAG','')}.png")
