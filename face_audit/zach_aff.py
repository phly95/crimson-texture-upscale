"""zach_aff.py SEEDS: affine paste of the flat emblem fitted to (a) the 4 baseline end points of the user's guide lines and (b) the oval outline (texture ball circle -> traced outline, nearest-point); klein then wraps it.
env TAG, WO (outline weight)"""
import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit"); sys.path.insert(0, "/home/deck/workspace/csvm-hd/tools")
import numpy as np, cv2
from PIL import Image
from scipy.optimize import least_squares
import zach_guides as G
Z = "/home/deck/workspace/csvm-hd/face_audit/zach/"; TAG = os.environ.get("TAG", "af"); WO = float(os.environ.get("WO", "1.0"))
A = [G.window(p) for p in G.lines_in_anchor()]; OUT = G.window(G.outline_in_anchor())
f0, f1 = np.array([110, 452]) * 1.6, np.array([470, 458]) * 1.6; h0, h1 = np.array([205, 585]) * 1.6, np.array([545, 530]) * 1.6
tex_pts = np.array([f0, f1, h0, h1]); dst_pts = np.array([A[0][0], A[0][-1], A[1][0], A[1][-1]])
TC = np.array([325.0, 255.0]) * 1.6; TR = 250.0 * 1.6
ang = np.linspace(0, 2 * np.pi, 120, endpoint=False); CIRC = TC + TR * np.c_[np.cos(ang), np.sin(ang)]
def apply(p, P): M = np.array([[p[0], p[1], p[4]], [p[2], p[3], p[5]]]); return P @ M[:, :2].T + M[:, 2]
def resid(p):
    r1 = (apply(p, tex_pts) - dst_pts).ravel() / 25.0
    c = apply(p, CIRC); d = np.linalg.norm(OUT[:, None, :] - c[None, :, :], axis=2).min(1)       # every traced outline point must lie on the projected circle
    return np.r_[r1, WO * d / 25.0]
p0 = np.linalg.lstsq(np.c_[tex_pts, np.ones(4)], dst_pts, rcond=None)[0]; p0 = [p0[0, 0], p0[1, 0], p0[0, 1], p0[1, 1], p0[2, 0], p0[2, 1]]
r = least_squares(resid, p0); p = r.x; print("affine", np.round(p, 3), "baseline err px", np.round(np.linalg.norm(apply(p, tex_pts) - dst_pts, axis=1), 1))
M = np.array([[p[0], p[1], p[4]], [p[2], p[3], p[5]]], np.float32)
em = np.asarray(Image.open(Z + "gypsy_hd.png").convert("RGBA"), np.float32) / 255; em[..., :3] = em[..., :3].mean(-1, keepdims=True)
wp = cv2.warpAffine(em, M, (1024, 1024), flags=cv2.INTER_AREA, borderValue=(0, 0, 0, 0))
ref = np.asarray(Image.open(Z + "ref.png").convert("RGB"), np.float32)
old = np.zeros((1024, 1024), np.float32); cv2.ellipse(old, (298, 448), (345, 178), 62, 0, 360, 1.0, -1)
old = cv2.GaussianBlur(old, (0, 0), 6)[..., None]; plate = ref * (1 - old) + ref.mean(-1, keepdims=True) * 0.30 * old
a = wp[..., 3:]; out = plate * (1 - a) + wp[..., :3] * 255 * a
Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(Z + f"{TAG}_in.png")
if len(sys.argv) > 1:
    sys.path.insert(0, "/home/deck/workspace/csvm-hd/tools"); import tile_upscale as tu
    P = ("In this black and white photo of an aircraft nose, a flat emblem (a crystal ball with a woman holding tarot cards and the script 'Fortune Hunters') has been laid over the fuselage, already slanted. "
         "Paint that emblem as a decal applied to the curved fuselage: it wraps around the curved metal like it is printed on a cylinder, with strong perspective foreshortening, a narrow slanted oval that follows the surface and curves away toward the upper right where the fuselage turns. "
         "Keep the position, size, outline and the slant of the two lettering lines exactly as laid out: 'Fortune' on the first line and 'Hunters' on the second line, both running down toward the lower right at the same angle. "
         "Keep the artwork and lettering of the emblem the same, only fitted onto the surface with matching light and film grain, and write the title only once. "
         "Remove everything of the old badge. Keep the sky, the gun barrel, the hand and the sleeve exactly as they are.")
    for sd in [int(x) for x in sys.argv[1:]]:
        tu.locked_gen(Z + f"{TAG}_in.png", Z + f"{TAG}_s{sd}.png", P, 1024, sd); print("seed", sd, flush=True)
