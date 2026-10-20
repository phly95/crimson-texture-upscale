"""headtex_cpdraw.py TARGET [steps]: hand-drawn (klein cartoon) pilot head. For each side view (front, q1, q2, right, left) the mesh render of the original (headtex/v_<tg>_<view>.png) is klein-redrawn,
then registered to the render by SILHOUETTE (similarity from the head masks, then a smooth flow on the mask distance fields) so klein's re-framing cannot cause doubling; finally one view per triangle is baked (argmax) over the
filtered texture (headtex/<tg>_toonf.png, which supplies back / top / neck) -> headtex/<tg>_draw_bake.png (RGBA; alpha = baked) and silhouette IoU per view."""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_bake as B, headtex_fin as F, headtex_run as R, headtex_cptoon as CT
T = F.T; D = R.D
VIEWS = ("front", "q1", "q2", "right", "left")
def head_mask(img, tol=14):
    """head = everything not connected to the plain grey background (flood from the corners); the render's background is (128,128,128)"""
    h, w = img.shape[:2]; bg = np.median(np.r_[img[:6, :6].reshape(-1, 3), img[:6, -6:].reshape(-1, 3), img[-6:, :6].reshape(-1, 3)], axis=0)
    diff = np.abs(img.astype(np.int16) - bg).max(-1) > tol; diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(diff); big = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]); m = (lab == big).astype(np.uint8)
    inv = 1 - m; n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv)
    m = m | np.isin(lab2, [i for i in range(1, n2) if st2[i, cv2.CC_STAT_AREA] < 0.01 * h * w]).astype(np.uint8)       # fill holes
    return m.astype(bool)
def moments(m):
    ys, xs = np.nonzero(m); return np.array([xs.mean(), ys.mean()]), np.sqrt(m.sum())
def register(gen, ref, gm, rm, smooth=12.0, clip=70.0):
    """gen -> ref frame: similarity from silhouette moments, then smooth DIS flow on blurred masks, then the usual grey-structure flow"""
    (cg, sg), (cr, sr) = moments(gm), moments(rm); k = sr / sg
    A = np.array([[k, 0, cr[0] - k * cg[0]], [0, k, cr[1] - k * cg[1]]], np.float32); h, w = ref.shape[:2]
    g1 = cv2.warpAffine(gen, A, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT); m1 = cv2.warpAffine(gm.astype(np.uint8) * 255, A, (w, h), flags=cv2.INTER_LINEAR)
    f = lambda m: cv2.GaussianBlur((m > 127).astype(np.float32) * 255 if m.dtype == np.uint8 else m.astype(np.float32) * 255, (0, 0), 6)
    u8 = lambda x: np.clip(x, 0, 255).astype(np.uint8)
    fl = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(u8(f(rm.astype(np.uint8) * 255)), u8(f(m1)), None)
    fl = cv2.GaussianBlur(fl, (0, 0), smooth); mg = np.hypot(fl[..., 0], fl[..., 1]); fl *= np.minimum(1, clip / np.maximum(mg, 1e-6))[..., None]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)); g2 = cv2.remap(g1, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    m2 = cv2.remap(m1, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR) > 127
    return g2, m2
def twin_skip(model, pack, name, xsign):
    """the atlas is a half head shared by both sides of the mesh: triangles on the x*xsign<0 side whose mirror twin uses the same UV spot are not baked (the twin, drawn from the kept side, fills those texels)"""
    V, T = B.HR.tris(model, pack); cs = {j: V[list(vi)].mean(0) for j, (vi, uv, nm) in enumerate(T) if nm == name}; ids = list(cs); arr = np.array([cs[j] for j in ids]); skip = []
    for j in ids:
        c = cs[j]
        if c[0] * xsign >= -1e-3: continue
        m = np.where(np.linalg.norm(arr - c * np.array([-1, 1, 1]), axis=1) < 1e-3)[0]
        if len(m) and np.abs(np.array(T[j][1]).mean(0) - np.array(T[ids[m[0]]][1]).mean(0)).max() < 0.02: skip.append(j)
    return skip
def neck_tris(model, pack, name, ymax=0.1):
    """the plain neck box: its drawn views disagree at the seams, so it stays with the filtered base"""
    V, T = B.HR.tris(model, pack); return [j for j, (vi, uv, nm) in enumerate(T) if nm == name and V[list(vi)][:, 1].max() < ymax]
def run(tg, steps="cb", seed=7, xsign=1):
    C = R.TARGETS[tg]; views = []
    for k in VIEWS:
        y, p = R.VIEWS[k]; src, out, fin = D / f"v_{tg}_{k}.png", D / f"gd_{tg}_{k}.png", D / f"rd_{tg}_{k}.png"
        if "c" in steps and not fin.exists():
            T.locked_gen_wh(str(src), str(out), "This is " + CT.WHO[tg] + CT.STYLE, 1024, 1024, seed)
            ref = np.array(Image.open(src).convert("RGB")); gen = np.array(Image.open(out).convert("RGB").resize((1024, 1024), Image.LANCZOS))
            rm, gm = head_mask(ref), head_mask(gen); g2, m2 = register(gen, ref, gm, rm)
            iou = (m2 & rm).sum() / (m2 | rm).sum(); w, fl = F.align(g2, ref, smooth=14.0, clip=30.0)
            fn = np.clip(T.colour_lock(w / 255., ref / 255., 10.0, 30.0), 0, 1); Image.fromarray((fn * 255 + .5).astype(np.uint8)).save(fin)
            print(tg, k, "silhouette IoU raw %.3f -> %.3f" % ((gm & rm).sum() / (gm | rm).sum(), iou), flush=True)
        if "b" in steps: views.append((y, p, np.array(Image.open(fin).convert("RGB")), np.load(D / f"v_{tg}_{k}_z.npy")))
    if "b" in steps:
        name, stem, (tw, th), _ = C["bake"][0]
        b, w = B.bake(views, model=C["model"], pack=C["pack"], only=name, TS=tw, TSH=th, pw=CT.PW, skip_extra=twin_skip(C['model'], C['pack'], name, xsign) + neck_tris(C['model'], C['pack'], name))
        a = (w > 1e-30).astype(np.uint8) * 255; Image.fromarray(np.dstack([np.clip(b + .5, 0, 255).astype(np.uint8), a]), "RGBA").save(D / f"{tg}_draw_bake{'' if xsign == 1 else '_m'}.png"); print(tg, "baked texels %.2f" % (a > 0).mean())
if __name__ == "__main__": run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "cb", xsign=int(sys.argv[3]) if len(sys.argv) > 3 else 1)
