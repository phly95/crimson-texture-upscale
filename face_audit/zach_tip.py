"""zach_tip.py [SEEDS]: start from the baseline/oval blended affine (WO 0.35) and add a local thin-plate correction only around the oval's pointed top-left tip, so the ball reaches the traced OG outline there
while the lettering stays on the user's baselines. Writes tip_in.png (+ klein seeds tip_s<seed>.png)"""
import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit"); sys.path.insert(0, "/home/deck/workspace/csvm-hd/tools")
import numpy as np, cv2
from PIL import Image
from scipy.optimize import least_squares
from scipy.interpolate import RBFInterpolator
import zach_guides as G
Z = "/home/deck/workspace/csvm-hd/face_audit/zach/"; TAG = os.environ.get("TAG", "tip")
A = [G.window(p) for p in G.lines_in_anchor()]; OUT = G.window(G.outline_in_anchor())
f0, f1 = np.array([110, 452]) * 1.6, np.array([470, 458]) * 1.6; h0, h1 = np.array([205, 585]) * 1.6, np.array([545, 530]) * 1.6
tex4 = np.array([f0, f1, h0, h1]); dst4 = np.array([A[0][0], A[0][-1], A[1][0], A[1][-1]])
TC = np.array([325.0, 255.0]) * 1.6; TR = 250.0 * 1.6
_al = (np.asarray(Image.open(Z + "gypsy_hd.png").convert("RGBA"))[..., 3] > 128).astype(np.uint8)
def radial(theta):
    r = np.arange(0, 700, 2.0); xs = (TC[0] + r * np.cos(theta)).astype(int); ys = (TC[1] + r * np.sin(theta)).astype(int)
    ok = (xs >= 0) & (xs < 1024) & (ys >= 0) & (ys < 1024); hit = np.zeros(len(r), bool); hit[ok] = _al[ys[ok], xs[ok]] > 0
    return r[np.where(hit)[0].max()] if hit.any() else TR
ang = np.linspace(0, 2 * np.pi, 120, endpoint=False); CIRC = TC + TR * np.c_[np.cos(ang), np.sin(ang)]
def apply(p, P): M = np.array([[p[0], p[1], p[4]], [p[2], p[3], p[5]]]); return P @ M[:, :2].T + M[:, 2]
def resid(p, wo=0.35):
    r1 = (apply(p, tex4) - dst4).ravel() / 25.0; c = apply(p, CIRC); d = np.linalg.norm(OUT[:, None, :] - c[None, :, :], axis=2).min(1); return np.r_[r1, wo * d / 25.0]
p0 = np.linalg.lstsq(np.c_[tex4, np.ones(4)], dst4, rcond=None)[0]; p0 = [p0[0, 0], p0[1, 0], p0[0, 1], p0[1, 1], p0[2, 0], p0[2, 1]]
p = least_squares(resid, p0).x
M2 = np.array([[p[0], p[1], p[4]], [p[2], p[3], p[5]]]); Minv = np.linalg.inv(np.vstack([M2, [0, 0, 1]]))[:2]      # window -> texture (affine)
aff_inv = lambda Q: Q @ Minv[:, :2].T + Minv[:, 2]
# correction landmarks: outline points near the tip (indices: the polyline starts at the tip and runs along the upper-right edge; the last points run down the left edge)
n = len(OUT); tipidx = [0, 1, 2, n - 1, n - 2]
corr_w = OUT[tipidx]; corr_t = []
for q in corr_w:
    t = aff_inv(q[None])[0]; d = t - TC; corr_t.append(TC + d / np.linalg.norm(d) * radial(np.arctan2(d[1], d[0])))
corr_t = np.array(corr_t)
# pinned points: a grid over the emblem + the baseline points; they keep the affine mapping
gx, gy = np.meshgrid(np.arange(60, 700, 70), np.arange(80, 800, 70)); pin_w = np.c_[gx.ravel(), gy.ravel()].astype(float)
dtip = np.min(np.linalg.norm(pin_w[:, None, :] - OUT[tipidx][None], axis=2), axis=1); pin_w = pin_w[dtip > 140]
pin_t = aff_inv(pin_w)
W = np.vstack([corr_w, pin_w]); T = np.vstack([corr_t, pin_t])
far = np.array([[-600, -600], [1600, -600], [-600, 1600], [1600, 1600]], float); farT = aff_inv(far)
SM = float(os.environ.get("SM", "30")); w = np.r_[np.ones(len(corr_w)), np.full(len(pin_w), 3.0)]
rbf = RBFInterpolator(np.vstack([W, far]), np.vstack([T, farT]), kernel="thin_plate_spline", smoothing=np.r_[SM / w ** 2, np.full(4, 1e6)])
yy, xx = np.mgrid[0:1024, 0:1024]; Pg = np.c_[xx.ravel(), yy.ravel()].astype(float); Tm = np.zeros((len(Pg), 2), np.float32)
for i in range(0, len(Pg), 200000): Tm[i:i + 200000] = rbf(Pg[i:i + 200000])
Tm = Tm.reshape(1024, 1024, 2)
em = np.asarray(Image.open(Z + "gypsy_hd.png").convert("RGBA"), np.float32) / 255; em[..., :3] = em[..., :3].mean(-1, keepdims=True)
wp = cv2.remap(em, Tm[..., 0], Tm[..., 1], cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
ref = np.asarray(Image.open(Z + "ref.png").convert("RGB"), np.float32)
old = np.zeros((1024, 1024), np.float32); cv2.ellipse(old, (298, 448), (345, 178), 62, 0, 360, 1.0, -1)
old = cv2.GaussianBlur(old, (0, 0), 6)[..., None]; plate = ref * (1 - old) + ref.mean(-1, keepdims=True) * 0.30 * old
a = wp[..., 3:]; out = plate * (1 - a) + wp[..., :3] * 255 * a
Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(Z + f"{TAG}_in.png"); print("composite ok")
if len(sys.argv) > 1:
    import tile_upscale as tu
    P = ("In this black and white photo of an aircraft nose, a flat emblem (a crystal ball with a woman holding tarot cards and the script 'Fortune Hunters') has been laid over the fuselage, already slanted and pulled into a pointed teardrop at its upper left. "
         "Paint that emblem as a decal applied to the curved fuselage: it wraps around the curved metal like it is printed on a cylinder, with strong perspective foreshortening, a narrow slanted teardrop oval that follows the surface, pointed at the upper left and curving away toward the upper right where the fuselage turns. "
         "Keep the position, size, outline (including the pointed upper-left end) and the slant of the two lettering lines exactly as laid out: 'Fortune' on the first line and 'Hunters' on the second line, both running down toward the lower right at the same angle. "
         "Keep the artwork and lettering of the emblem the same, only fitted onto the surface with matching light and film grain, and write the title only once. "
         "Remove everything of the old badge. Keep the sky, the gun barrel, the hand and the sleeve exactly as they are.")
    for sd in [int(x) for x in sys.argv[1:]]:
        tu.locked_gen(Z + f"{TAG}_in.png", Z + f"{TAG}_s{sd}.png", P, 1024, sd); print("seed", sd, flush=True)
