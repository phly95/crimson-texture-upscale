"""zach_apply.py SEED: put the crystal-ball emblem (klein pass SEED) into the anchor (3ad54545) and, with the anchor's head, into the other two copies of the photo.
writes face_audit/zach/final_<suffix>.png (RGBA, same size as the picture's current redraw)"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit")
import regen as R, numpy as np, cv2
from PIL import Image
from skimage import color as skc
NAME = sys.argv[1]; Z = "/home/deck/workspace/csvm-hd/face_audit/zach/"
f = lambda s: [u for u in R.last if u.endswith(s)][0]
ua = f("3ad54545"); ca = R.chosen(ua, "latest"); A0 = ca[..., :3].astype(np.float32) / 255
SC = 1024 / 540
# 1. anchor: new emblem window pasted in with a feathered ellipse around the badge and its lettering
win = np.asarray(Image.open(Z + NAME).convert("RGB").resize((540, 540), Image.LANCZOS), np.float32) / 255
Wf = np.zeros_like(A0); Wf[40:580, 0:540] = win
Ma = np.zeros((1024, 1024), np.float32); cv2.ellipse(Ma, (157, 276), (235, 140), 62, 0, 360, 1.0, -1)
Ma[:40] = 0; Ma[580:] = 0; Ma[:, 540:] = 0
Ma = cv2.GaussianBlur(Ma, (0, 0), 12)
A1 = A0 * (1 - Ma[..., None]) + Wf * Ma[..., None]
anc = np.dstack([A1, np.ones((1024, 1024), np.float32)])
Image.fromarray((anc * 255 + .5).astype(np.uint8), "RGBA").save(Z + "final_3ad54545.png")
# 2. the other copies: head already regenerated (head_<s>.png); emblem warped from the new anchor through the originals' homography
def apply(ut_sfx):
    ut = f(ut_sfx); base = np.asarray(Image.open(Z + f"head_{ut_sfx}.png").convert("RGBA"), np.float32) / 255
    Hog, n = R.homography(ua, ut); oa, ot = R.og(ua), R.og(ut)
    Sa = np.diag([1024 / oa.shape[1], 1024 / oa.shape[0], 1.0]); St = np.diag([base.shape[1] / ot.shape[1], base.shape[0] / ot.shape[0], 1.0])
    Ht = St @ Hog @ np.linalg.inv(Sa); Hh, Ww = base.shape[:2]
    S = cv2.warpPerspective(A1, Ht, (Ww, Hh), flags=cv2.INTER_AREA if abs(np.linalg.det(Ht[:2, :2])) < 1 else cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    M = cv2.warpPerspective(Ma, Ht, (Ww, Hh), flags=cv2.INTER_LINEAR, borderValue=0.0)
    old = base[..., :3]
    # never over the frame / key colour
    gt = ((old[..., 1] - np.maximum(old[..., 0], old[..., 2])) > 0.25).astype(np.uint8)
    wt = cv2.morphologyEx((old.min(-1) > 0.93).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nn, lb, st, _ = cv2.connectedComponentsWithStats(wt); wb = np.zeros_like(wt)
    for i in range(1, nn):
        if st[i, cv2.CC_STAT_AREA] > 0.01 * Ww * Hh: wb[lb == i] = 1
    prot = cv2.GaussianBlur(cv2.dilate(np.maximum(gt, wb), np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 2)
    M = M * (1 - prot)
    # broad tone of the badge to the plane's tone just outside it (lightness and colour), measured on a ring
    lo, ls = skc.rgb2lab(old), skc.rgb2lab(S)
    hard = (M > 0.5).astype(np.uint8); ring = cv2.dilate(hard, np.ones((61, 61), np.uint8)) - cv2.dilate(hard, np.ones((15, 15), np.uint8))
    ring = ring.astype(np.float32) * (1 - cv2.dilate(np.maximum(gt, wb), np.ones((41, 41), np.uint8)).astype(np.float32))
    inn = cv2.erode(hard, np.ones((9, 9), np.uint8)).astype(np.float32)
    def mean(x, w): return (x * w[..., None]).reshape(-1, 3).sum(0) / (w.sum() + 1e-6)
    dm = mean(lo, ring) - mean(ls, inn)
    ls[..., 1:] += dm[1:]; ls[..., 0] += dm[0] * 0.5
    S2 = np.clip(skc.lab2rgb(ls), 0, 1).astype(np.float32)
    res = old * (1 - M[..., None]) + S2 * M[..., None]
    out = base.copy(); out[..., :3] = np.where((base[..., 3] < 0.999)[..., None], old, res)
    Image.fromarray((out * 255 + .5).astype(np.uint8), "RGBA").save(Z + f"final_{ut_sfx}.png"); print(ut_sfx, "ok", n)
for t in ("6b4b3dd2", "d947fbd6"): apply(t)
