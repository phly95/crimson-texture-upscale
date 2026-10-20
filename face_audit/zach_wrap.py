"""zach_wrap.py: fit the wrapped placement of the 48gypsy texture on the fuselage from landmark pairs (OG grid of the 6b4b3dd2 crop <-> texture), render the HD texture through a thin-plate warp.
  zach_wrap.py check   -> overlay of the warped texture on the OG crop (6x) and on the anchor
  zach_wrap.py render  -> face_audit/zach/wrap_layer.png (RGBA 1024, anchor result space, greyscale)"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit")
import numpy as np, cv2, regen as R
from PIL import Image
from scipy.interpolate import RBFInterpolator
Z = "/home/deck/workspace/csvm-hd/face_audit/zach/"
# (OG grid x,y of the 6x crop (50,75)+...) <-> (texture view x,y of 640 px)
LM = [((75, 310), (100, 400)), ((385, 565), (470, 440)), ((130, 545), (210, 545)), ((415, 690), (545, 510)), ((300, 430), (345, 345)),
      ((300, 705), (330, 625)), ((95, 75), (320, 12)), ((470, 470), (580, 290)), ((62, 250), (70, 300))]
import json, os
if os.path.exists(Z + "lm.json"): LM = [tuple(map(tuple, p)) for p in json.load(open(Z + "lm.json"))]
src = np.array([a for a, b in LM], float); dst = np.array([b for a, b in LM], float)
# affine fit (grid -> texture) to stabilise the far field
Aff, *_ = np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)
res = np.c_[src, np.ones(len(src))] @ Aff - dst; print("affine residual px (texture view):", np.round(np.abs(res).max(0), 1))
far = np.array([[-300, -300], [1000, -300], [-300, 1200], [1000, 1200]], float); farT = np.c_[far, np.ones(4)] @ Aff
rbf = RBFInterpolator(np.r_[src, far], np.r_[dst, farT], kernel="thin_plate_spline", smoothing=float(os.environ.get("SM", "30")))
f = lambda s: [u for u in R.last if u.endswith(s)][0]
ua, ut = f("3ad54545"), f("6b4b3dd2")
Hog, n = R.homography(ua, ut)                         # anchor og -> 6b og
def dest_to_tex(P):                                   # P: Nx2 anchor-result px -> texture HD px
    og = P / 4.0; h = np.c_[og, np.ones(len(og))] @ Hog.T; og6 = h[:, :2] / h[:, 2:3]
    g = (og6 - np.array([50, 75])) * 6
    return rbf(g) * 1.6
tex = np.asarray(Image.open(Z + "gypsy_hd.png").convert("RGBA"), np.float32) / 255
def layer(box=(0, 40, 560, 600)):
    x0, y0, x1, y1 = box; yy, xx = np.mgrid[y0:y1, x0:x1]; P = np.c_[xx.ravel(), yy.ravel()].astype(float)
    T = dest_to_tex(P).reshape(y1 - y0, x1 - x0, 2).astype(np.float32)
    wp = cv2.remap(tex, T[..., 0], T[..., 1], cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    out = np.zeros((1024, 1024, 4), np.float32); out[y0:y1, x0:x1] = wp; return out
L = layer(); L[..., :3] = L[..., :3].mean(-1, keepdims=True)
if sys.argv[1] == "check":
    anc = np.asarray(Image.open(Z + "final_3ad54545.png").convert("RGB"), np.float32) / 255
    old = np.asarray(R.chosen(ua, "latest")[..., :3], np.float32) / 255
    a = L[..., 3:]; ov = old * (1 - 0.65 * a) + L[..., :3] * 0.65 * a
    Image.fromarray((ov[30:620, 0:600] * 255).astype(np.uint8)).resize((900, 885)).save(Z + "wrap_check.png")
    # same on the OG 6b crop
    o6 = np.asarray(Image.open("/home/deck/workspace/csvm-hd/pipeline/units/scrap_MS_P_ZACHARYANDPLANE__6b4b3dd2/orig.png").convert("RGB").crop((50, 75, 170, 215)).resize((720, 840), Image.BICUBIC), np.float32) / 255
    Image.fromarray((o6 * 255).astype(np.uint8)).save(Z + "wrap_og.png")
else:
    Image.fromarray((L * 255 + .5).astype(np.uint8), "RGBA").save(Z + "wrap_layer.png"); print("ok")
if sys.argv[1] == "check":
    gy, gx = np.mgrid[0:840, 0:720]; G = np.c_[gx.ravel(), gy.ravel()].astype(float)
    T = (rbf(G) * 1.6).reshape(840, 720, 2).astype(np.float32)
    wp = cv2.remap(tex, T[..., 0], T[..., 1], cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    wp[..., :3] = wp[..., :3].mean(-1, keepdims=True); a = wp[..., 3:]
    o6 = np.asarray(Image.open(Z + "wrap_og.png"), np.float32) / 255
    ov = o6 * (1 - 0.6 * a) + wp[..., :3] * 0.6 * a
    side = np.concatenate([o6, ov], 1); Image.fromarray((side * 255).astype(np.uint8)).save(Z + "wrap_og_check.png")
    for (s_, d_) in LM: print(s_, "->", np.round(rbf(np.array([s_], float))[0], 0), "want", d_, " affine", np.round(np.r_[s_, 1] @ Aff, 0))
