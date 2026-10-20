"""zach_tps2.py: thin-plate warp of the flat 48gypsy fitted to the user's measurements: both baselines (guide polylines) and the oval outline -> face_audit/zach/<TAG>_in.png (window composite). SM smoothing"""
import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit"); sys.path.insert(0, "/home/deck/workspace/csvm-hd/tools")
import numpy as np, cv2
from PIL import Image
from scipy.interpolate import RBFInterpolator
import zach_guides as G
Z = "/home/deck/workspace/csvm-hd/face_audit/zach/"; TAG = os.environ.get("TAG", "tp"); SM = float(os.environ.get("SM", "200")); WO = float(os.environ.get("WO", "1"))
A = [G.window(p) for p in G.lines_in_anchor()]; OUT = G.window(G.outline_in_anchor())
def resample(P, n):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; t = np.linspace(0, d[-1], n)
    return np.c_[np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])]
T1 = np.array([(110, 452), (200, 454), (290, 456), (380, 457), (470, 458)]) * 1.6; T2 = np.array([(205, 585), (300, 565), (390, 550), (470, 540), (545, 530)]) * 1.6
W = list(resample(A[0], 5)) + list(resample(A[1], 5)); Tx = list(T1) + list(T2)
# outline: window outline point -> circle point (via the affine fitted to baselines only, radial projection)
W4 = np.array([A[0][0], A[0][-1], A[1][0], A[1][-1]]); T4 = np.array([T1[0], T1[-1], T2[0], T2[-1]])
Aff = np.linalg.lstsq(np.c_[T4, np.ones(4)], W4, rcond=None)[0]                                        # texture -> window
Ainv = np.linalg.lstsq(np.c_[W4, np.ones(4)], T4, rcond=None)[0]                                      # window -> texture
TC = np.array([325.0, 255.0]) * 1.6; TR = 250.0 * 1.6
_al = (np.asarray(Image.open(Z + "gypsy_hd.png").convert("RGBA"))[..., 3] > 128).astype(np.uint8)
def radial(theta):                                              # farthest opaque texel along the ray from the ball centre (HD px)
    r = np.arange(0, 700, 2.0); xs = (TC[0] + r * np.cos(theta)).astype(int); ys = (TC[1] + r * np.sin(theta)).astype(int)
    ok = (xs >= 0) & (xs < 1024) & (ys >= 0) & (ys < 1024); hit = np.zeros(len(r), bool); hit[ok] = _al[ys[ok], xs[ok]] > 0
    return r[np.where(hit)[0].max()] if hit.any() else TR
src_out = []; dst_out = []
for q in resample(np.vstack([OUT, OUT[:1]]), 24)[:-1]:
    t = np.r_[q, 1] @ Ainv; d = t - TC; c = TC + d / np.linalg.norm(d) * radial(np.arctan2(d[1], d[0])); src_out.append(q); dst_out.append(c)
W = np.array(W + src_out); Tx = np.array(Tx + dst_out)
far = np.array([[-500, -500], [1500, -500], [-500, 1500], [1500, 1500]], float); farT = np.c_[far, np.ones(4)] @ Ainv
wt = np.r_[np.ones(10), np.full(len(src_out), WO)]
rbf = RBFInterpolator(np.r_[W, far], np.r_[Tx, farT], kernel="thin_plate_spline", smoothing=np.r_[SM / wt ** 2, np.full(4, 1e6)])
gy, gx = np.mgrid[0:1024, 0:1024]; P = np.c_[gx.ravel(), gy.ravel()].astype(float)
Tm = np.zeros((1024 * 1024, 2), np.float32)
for i in range(0, len(P), 200000): Tm[i:i + 200000] = rbf(P[i:i + 200000])
Tm = Tm.reshape(1024, 1024, 2)
em = np.asarray(Image.open(Z + "gypsy_hd.png").convert("RGBA"), np.float32) / 255; em[..., :3] = em[..., :3].mean(-1, keepdims=True)
wp = cv2.remap(em, Tm[..., 0], Tm[..., 1], cv2.INTER_AREA if False else cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
ref = np.asarray(Image.open(Z + "ref.png").convert("RGB"), np.float32)
old = np.zeros((1024, 1024), np.float32); cv2.ellipse(old, (298, 448), (345, 178), 62, 0, 360, 1.0, -1)
old = cv2.GaussianBlur(old, (0, 0), 6)[..., None]; plate = ref * (1 - old) + ref.mean(-1, keepdims=True) * 0.30 * old
a = wp[..., 3:]; out = plate * (1 - a) + wp[..., :3] * 255 * a
Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(Z + f"{TAG}_in.png")
Image.fromarray((np.dstack([wp[..., :3].mean(-1)] * 3 + [wp[..., 3]]) * 255).astype(np.uint8), "RGBA").save(Z + f"{TAG}_layer.png")
print("ok")
