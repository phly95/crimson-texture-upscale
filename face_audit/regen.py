"""regen.py: replace a whole head (face + hair) in a redraw by the same head from the best redraw of that photo, registered through the two ORIGINALS (SIFT homography)."""
import sys, json, re, warnings
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from skimage import color as skc
warnings.filterwarnings("ignore")
ROOT = Path("/home/deck/workspace/csvm-hd"); sys.path.insert(0, str(ROOT / "face_audit")); sys.path.insert(0, str(ROOT / "tools"))
import transfer as T, export_winners as ew
last = {}
for l in open(ROOT / "pipeline/reviews.jsonl"):
    if l.strip():
        r = json.loads(l); last[r["uid"]] = r
def latest_file(u):
    cj = ROOT / "pipeline/units" / u / "custom" / "custom.json"
    if cj.exists():
        items = json.load(open(cj)).get("items", [])
        if items: return ROOT / "pipeline/units" / u / "custom" / items[-1]["file"]
    return None
def chosen(u, base="chosen"):
    f = latest_file(u) if base == "latest" else None
    return np.asarray(Image.open(f or ew.get_chosen_file(u, last[u]["chosen"])).convert("RGBA"))
def og(u): return np.asarray(Image.open(ROOT / "pipeline/units" / u / "orig.png").convert("RGB"))
def homography(ua, ut):
    A, B = og(ua), og(ut)
    def prep(im):
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY); s = 800 / max(g.shape); return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), s
    ga, sa = prep(A); gb, sb = prep(B)
    sift = cv2.SIFT_create(4000); ka, da = sift.detectAndCompute(ga, None); kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10: return None, 0
    m = cv2.BFMatcher().knnMatch(da, db, k=2); good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 12: return None, len(good)
    pa = np.float32([ka[g.queryIdx].pt for g in good]); pb = np.float32([kb[g.trainIdx].pt for g in good])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if H is None or inl.sum() < 12: return None, int(inl.sum()) if inl is not None else 0
    Sa = np.diag([sa, sa, 1.0]); Sb = np.diag([sb, sb, 1.0])
    Hog = np.linalg.inv(Sb) @ H @ Sa                      # anchor-og px -> target-og px
    return Hog, int(inl.sum())
def regen(ua, ut, centre, k=(1.5, 1.7), feather=0.32, base="latest", anchor_base="latest", keep_thin=False, shift=-0.04):
    ca, ct = chosen(ua, anchor_base), chosen(ut, base)
    Hog, n = homography(ua, ut)
    if Hog is None: return None, f"no registration ({n} matches)"
    oa, ot = og(ua), og(ut)
    Sa = np.diag([ca.shape[1] / oa.shape[1], ca.shape[0] / oa.shape[0], 1.0]); St = np.diag([ct.shape[1] / ot.shape[1], ct.shape[0] / ot.shape[0], 1.0])
    Ht = St @ Hog @ np.linalg.inv(Sa)
    Hh, Ww = ct.shape[:2]
    S = cv2.warpPerspective(ca[..., :3].astype(np.float32) / 255, Ht, (Ww, Hh), flags=cv2.INTER_AREA if abs(np.linalg.det(Ht[:2, :2])) < 1 else cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    ac = ca.astype(np.float32) / 255
    a_green = ((ac[..., 1] - np.maximum(ac[..., 0], ac[..., 2])) > 0.2).astype(np.uint8)
    a_white = cv2.morphologyEx((ac[..., :3].min(-1) > 0.9).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(a_white); a_big = np.zeros_like(a_white)
    for i in range(1, n2):
        if st2[i, cv2.CC_STAT_AREA] > 0.01 * ac.shape[0] * ac.shape[1]: a_big[lab2 == i] = 1
    a_frame = np.maximum(np.maximum(a_green, a_big), (ac[..., 3] < 0.99).astype(np.uint8))
    a_ok = cv2.erode((1 - cv2.dilate(a_frame, np.ones((15, 15), np.uint8))).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
    valid = cv2.warpPerspective(a_ok, Ht, (Ww, Hh), flags=cv2.INTER_LINEAR, borderValue=0.0)
    valid = cv2.erode((valid > 0.999).astype(np.uint8), np.ones((9, 9), np.uint8))
    oldc = ct[..., :3].astype(np.float32) / 255
    green = ((oldc[..., 1] - np.maximum(oldc[..., 0], oldc[..., 2])) > 0.25).astype(np.uint8)       # the key colour of the picture's transparent surround
    tw = cv2.morphologyEx((oldc.min(-1) > 0.93).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n3, lab3, st3, _ = cv2.connectedComponentsWithStats(tw); tbig = np.zeros_like(tw)
    for i in range(1, n3):
        if st3[i, cv2.CC_STAT_AREA] > 0.01 * Ww * Hh: tbig[lab3 == i] = 1
    protect = cv2.GaussianBlur(cv2.dilate(np.maximum(green, tbig), np.ones((61, 61), np.uint8)).astype(np.float32), (0, 0), 8)
    faces = [f for f in T.detect(ct[..., :3]) if f[2] >= 40]
    f = min(faces, key=lambda f: np.hypot((f[0] + f[2] / 2) / Ww - centre[0], (f[1] + f[3] / 2) / Hh - centre[1]))
    cx, cy, w, h = f[0] + f[2] / 2, f[1] + f[3] / 2 + shift * f[3], f[2], f[3]
    m = np.zeros((Hh, Ww), np.float32); cv2.ellipse(m, (int(cx), int(cy)), (int(k[0] * w), int(k[1] * h)), 0, 0, 360, 1.0, -1)
    mo = cv2.GaussianBlur(m, (0, 0), feather * w) * valid.astype(np.float32) * (1 - protect)
    mi = cv2.GaussianBlur(cv2.erode(m, np.ones((3, 3), np.uint8), iterations=max(1, int(0.5 * k[0] * w))), (0, 0), 0.25 * w)
    A = ct.astype(np.float32) / 255; old = A[..., :3]
    ys, xs = np.where(mo > 0.002); y0, y1, x0, x1 = max(0, ys.min() - 2), min(Hh, ys.max() + 3), max(0, xs.min() - 2), min(Ww, xs.max() + 3)
    o, s = old[y0:y1, x0:x1], S[y0:y1, x0:x1]
    lo, ls = skc.rgb2lab(o), skc.rgb2lab(s); sig = 0.3 * w
    ls[..., 1:] = ls[..., 1:] - cv2.GaussianBlur(ls[..., 1:], (0, 0), sig) + cv2.GaussianBlur(lo[..., 1:], (0, 0), sig)
    dl = cv2.GaussianBlur(lo[..., :1], (0, 0), sig)[..., None] - cv2.GaussianBlur(ls[..., :1], (0, 0), sig)[..., None]
    ls[..., :1] = ls[..., :1] + dl * (1 - mi[y0:y1, x0:x1])[..., None]
    s2 = np.clip(skc.lab2rgb(ls), 0, 1).astype(np.float32)
    res = T.lap_blend(o.astype(np.float32), s2, mo[y0:y1, x0:x1].astype(np.float32), levels=5)
    new = A[..., :3].copy(); new[y0:y1, x0:x1] = np.clip(res, 0, 1)
    if keep_thin:                                              # thin bright vertical structures of the target (a rope, a pole) stay as they were
        L = cv2.cvtColor((old * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        op = cv2.morphologyEx(L, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, int(0.12 * w)) | 1, 1)))
        thin = np.clip((L - op - 0.08) / 0.12, 0, 1) * (mo > 0.05)
        thin = cv2.GaussianBlur(cv2.dilate(thin, np.ones((3, 3), np.uint8)), (0, 0), 1.0)[..., None]
        new = new * (1 - thin) + old * thin
    keep = (A[..., 3] < 0.999)[..., None]                       # never touch the colour under transparency
    A[..., :3] = np.where(keep, old, new)
    return (A * 255 + .5).astype(np.uint8), f"ok ({n} inliers)"


def recompose(ua, ut, base="chosen", anchor_base="chosen", erode=10, feather=5.0, tone_sigma=60.0):
    """the whole photo content of the target := the anchor's redraw of the same photo, warped through the originals' homography.
    The target keeps its frame/border/transparency (alpha, green key, near-white surround); tone is matched to the target's own broad tone."""
    ca, ct = chosen(ua, anchor_base), chosen(ut, base)
    Hog, n = homography(ua, ut)
    if Hog is None: return None, f"no registration ({n})"
    oa, ot = og(ua), og(ut)
    Sa = np.diag([ca.shape[1] / oa.shape[1], ca.shape[0] / oa.shape[0], 1.0]); St = np.diag([ct.shape[1] / ot.shape[1], ct.shape[0] / ot.shape[0], 1.0])
    Ht = St @ Hog @ np.linalg.inv(Sa); Hh, Ww = ct.shape[:2]
    S = cv2.warpPerspective(ca[..., :3].astype(np.float32) / 255, Ht, (Ww, Hh), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    ac = ca.astype(np.float32) / 255                                                   # the anchor's own surround (green key, paper border, transparency) is not photo content
    a_green = ((ac[..., 1] - np.maximum(ac[..., 0], ac[..., 2])) > 0.2).astype(np.uint8)
    a_white = cv2.morphologyEx((ac[..., :3].min(-1) > 0.9).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(a_white); a_big = np.zeros_like(a_white)
    for i in range(1, n2):
        if st2[i, cv2.CC_STAT_AREA] > 0.01 * ac.shape[0] * ac.shape[1]: a_big[lab2 == i] = 1
    a_frame = np.maximum(np.maximum(a_green, a_big), (ac[..., 3] < 0.99).astype(np.uint8))
    a_ok = cv2.erode((1 - cv2.dilate(a_frame, np.ones((7, 7), np.uint8))).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    valid = cv2.warpPerspective(a_ok, Ht, (Ww, Hh), flags=cv2.INTER_LINEAR, borderValue=0.0)
    valid = cv2.erode((valid > 0.999).astype(np.uint8), np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)).astype(np.float32)
    A = ct.astype(np.float32) / 255; old = A[..., :3]
    green = ((old[..., 1] - np.maximum(old[..., 0], old[..., 2])) > 0.25).astype(np.uint8)
    white = (old.min(-1) > 0.93).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))          # the paper border, not white clothes (those are small / inside the valid area)
    n_, lab, st, _ = cv2.connectedComponentsWithStats(white)
    big = np.zeros_like(white)
    for i in range(1, n_):
        if st[i, cv2.CC_STAT_AREA] > 0.01 * Ww * Hh: big[lab == i] = 1
    frame = np.maximum(green, big)
    m = valid * (1 - cv2.dilate(frame, np.ones((5, 5), np.uint8)).astype(np.float32)) * (A[..., 3] > 0.99)
    m = cv2.GaussianBlur(m, (0, 0), feather)
    # broad tone of the redraw -> broad tone of the target photo (inside the mask)
    w = (m > 0.9).astype(np.float32)[..., None]
    lo, ls = skc.rgb2lab(old), skc.rgb2lab(S)
    ws = cv2.GaussianBlur(w[..., 0], (0, 0), tone_sigma) + 1e-4
    def blur(x): return np.dstack([cv2.GaussianBlur(x[..., c] * w[..., 0], (0, 0), tone_sigma) / ws for c in range(x.shape[2])])
    ls[..., :1] += (blur(lo[..., :1]) - blur(ls[..., :1])) * 0.8
    ls[..., 1:] += (blur(lo[..., 1:]) - blur(ls[..., 1:]))
    S2 = np.clip(skc.lab2rgb(ls), 0, 1).astype(np.float32)
    res = old * (1 - m[..., None]) + S2 * m[..., None]
    A[..., :3] = np.where((A[..., 3] < 0.999)[..., None], old, res)
    return (A * 255 + .5).astype(np.uint8), f"ok ({n} inliers, {100 * (m > 0.5).mean():.0f}% of the picture replaced)"


def patch_region(ua, ut, box, base="latest", anchor_base="latest", feather=18.0, tone_sigma=40.0):
    """target pixels inside BOX (fractions x0,y0,x1,y1 of the target) := the anchor's redraw of the same photo, warped through the originals' homography, tone-matched at the border"""
    ca, ct = chosen(ua, anchor_base), chosen(ut, base)
    Hog, n = homography(ua, ut)
    if Hog is None: return None, f"no registration ({n})"
    oa, ot = og(ua), og(ut)
    Sa = np.diag([ca.shape[1] / oa.shape[1], ca.shape[0] / oa.shape[0], 1.0]); St = np.diag([ct.shape[1] / ot.shape[1], ct.shape[0] / ot.shape[0], 1.0])
    Ht = St @ Hog @ np.linalg.inv(Sa); Hh, Ww = ct.shape[:2]
    S = cv2.warpPerspective(ca[..., :3].astype(np.float32) / 255, Ht, (Ww, Hh), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpPerspective(np.ones(ca.shape[:2], np.float32), Ht, (Ww, Hh), flags=cv2.INTER_LINEAR, borderValue=0.0)
    valid = cv2.erode((valid > 0.999).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(np.float32)
    x0, y0, x1, y1 = [int(v * s) for v, s in zip(box, (Ww, Hh, Ww, Hh))]
    m = np.zeros((Hh, Ww), np.float32); m[y0:y1, x0:x1] = 1
    m = cv2.GaussianBlur(m, (0, 0), feather) * valid
    A = ct.astype(np.float32) / 255; old = A[..., :3]
    gt = ((old[..., 1] - np.maximum(old[..., 0], old[..., 2])) > 0.25).astype(np.uint8)                   # never paint over the key-colour surround or the frame
    wt = cv2.morphologyEx((old.min(-1) > 0.93).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nn, lb, stt, _ = cv2.connectedComponentsWithStats(wt); wbig = np.zeros_like(wt)
    for i in range(1, nn):
        if stt[i, cv2.CC_STAT_AREA] > 0.01 * Ww * Hh: wbig[lb == i] = 1
    prot = cv2.GaussianBlur(cv2.dilate(np.maximum(gt, wbig), np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 2)
    m = m * (1 - prot)
    lo, ls = skc.rgb2lab(old), skc.rgb2lab(S)
    pw = cv2.dilate(np.maximum(gt, wbig), np.ones((41, 41), np.uint8)).astype(np.float32)
    w = np.clip(1 - m, 0, 1) * (cv2.dilate(m, np.ones((61, 61), np.uint8)) > 0) * (1 - pw)      # ring of target pixels just outside the patch, away from the frame and the key colour
    ws = cv2.GaussianBlur(w, (0, 0), tone_sigma) + 1e-4
    def blur(x): return np.dstack([cv2.GaussianBlur(x[..., c] * w, (0, 0), tone_sigma) / ws for c in range(x.shape[2])])
    ls[..., 1:] += blur(lo[..., 1:]) - blur(ls[..., 1:])
    S2 = np.clip(skc.lab2rgb(ls), 0, 1).astype(np.float32)
    res = old * (1 - m[..., None]) + S2 * m[..., None]
    A[..., :3] = np.where((A[..., 3] < 0.999)[..., None], old, res)
    return (A * 255 + .5).astype(np.uint8), f"ok ({n} inliers)"
