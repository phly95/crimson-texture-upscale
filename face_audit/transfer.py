"""transfer.py: carry a face from the best redraw of a photo into its other redraws (landmark-aligned, pyramid-blended, chroma matched to the target)."""
import sys, json, re, collections, warnings
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from skimage import color as skc
warnings.filterwarnings("ignore")
ROOT = Path("/home/deck/workspace/csvm-hd"); sys.path.insert(0, str(ROOT / "tools"))
import export_winners as ew
M = ROOT / "face_audit/models"
det = cv2.FaceDetectorYN.create(str(M / "yunet.onnx"), "", (320, 320), 0.5, 0.3, 5000)
rec = cv2.FaceRecognizerSF.create(str(M / "sface.onnx"), "")

def detect(img_rgb):
    im = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR); h, w = im.shape[:2]
    sc = min(1.0, 1600 / max(h, w)); d = cv2.resize(im, None, fx=sc, fy=sc) if sc < 1 else im
    det.setInputSize((d.shape[1], d.shape[0])); _, f = det.detect(d)
    out = []
    for r in (f if f is not None else []):
        r = r.copy(); r[:14] /= sc; out.append(r)
    return out

def emb(img_rgb, r):
    im = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR); e = rec.feature(rec.alignCrop(im, r)).flatten(); return e / np.linalg.norm(e)

def lap_blend(a, b, m, levels=5):
    ga, gb, gm = [a], [b], [m]
    for _ in range(levels):
        ga.append(cv2.pyrDown(ga[-1])); gb.append(cv2.pyrDown(gb[-1])); gm.append(cv2.pyrDown(gm[-1]))
    out = ga[-1] * (1 - gm[-1][..., None]) + gb[-1] * gm[-1][..., None]
    for i in range(levels, 0, -1):
        sz = (ga[i - 1].shape[1], ga[i - 1].shape[0])
        la = ga[i - 1] - cv2.pyrUp(ga[i], dstsize=sz); lb = gb[i - 1] - cv2.pyrUp(gb[i], dstsize=sz); mk = gm[i - 1][..., None]
        out = cv2.pyrUp(out, dstsize=sz) + la * (1 - mk) + lb * mk
    return out

MAX_RESID = 0.12

def transfer(src_rgb, s_face, tgt_rgba, t_face, k_el=(1.35, 1.8)):
    """src_rgb uint8 HxWx3 with YuNet row s_face; tgt_rgba uint8 HxWx4 with row t_face -> new RGBA"""
    A = tgt_rgba.astype(np.float32) / 255; H, W = A.shape[:2]
    s_lm = np.array(s_face[4:14], np.float32).reshape(5, 2); t_lm = np.array(t_face[4:14], np.float32).reshape(5, 2)
    Mx, _ = cv2.estimateAffinePartial2D(s_lm, t_lm, method=cv2.LMEDS)
    scale = float(np.hypot(Mx[0, 0], Mx[1, 0]))
    d0 = float(np.linalg.norm(t_lm[3:].mean(0) - t_lm[:2].mean(0)))
    resid = float(np.sqrt(((s_lm @ Mx[:, :2].T + Mx[:, 2] - t_lm) ** 2).sum(1).mean()) / max(d0, 1e-6))
    if resid > MAX_RESID or not (0.4 < scale < 2.5):
        return None, (scale, resid)
    S = cv2.warpAffine(src_rgb.astype(np.float32) / 255, Mx, (W, H), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    ec = t_lm[:2].mean(0); mc = t_lm[3:].mean(0); d = float(np.linalg.norm(mc - ec)); c = (ec + mc) / 2
    ang = np.degrees(np.arctan2(*(t_lm[1] - t_lm[0])[::-1]))
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, (int(round(c[0])), int(round(c[1]))), (int(k_el[0] * d), int(k_el[1] * d)), ang, 0, 360, 1.0, -1)
    mo = cv2.GaussianBlur(m, (0, 0), 0.28 * d)
    mi = cv2.GaussianBlur(cv2.erode(m, np.ones((3, 3), np.uint8), iterations=max(1, int(0.45 * k_el[0] * d))), (0, 0), 0.22 * d)
    old = A[..., :3]
    ys, xs = np.where(mo > 0.002); y0, y1, x0, x1 = max(0, ys.min() - 2), min(H, ys.max() + 3), max(0, xs.min() - 2), min(W, xs.max() + 3)
    o, s = old[y0:y1, x0:x1], S[y0:y1, x0:x1]
    lo, ls = skc.rgb2lab(o), skc.rgb2lab(s)
    sig = 0.36 * d
    ab_t = cv2.GaussianBlur(lo[..., 1:], (0, 0), sig); ab_s = cv2.GaussianBlur(ls[..., 1:], (0, 0), sig)
    ls[..., 1:] = ls[..., 1:] - ab_s + ab_t                                    # keep the source's fine chroma, take the target's broad colour
    dl = cv2.GaussianBlur(lo[..., :1], (0, 0), sig)[..., None] - cv2.GaussianBlur(ls[..., :1], (0, 0), sig)[..., None]
    ls[..., :1] = ls[..., :1] + dl * (1 - mi[y0:y1, x0:x1])[..., None]          # luminance matched near the border only
    s2 = np.clip(skc.lab2rgb(ls), 0, 1).astype(np.float32)
    out = lap_blend(o.astype(np.float32), s2, mo[y0:y1, x0:x1].astype(np.float32), levels=5)
    A[y0:y1, x0:x1, :3] = np.clip(out, 0, 1)
    return (A * 255 + .5).astype(np.uint8), (scale, resid)

def match(anchor_faces, faces, W_a, H_a, W_t, H_t, tol=0.09):
    pairs = []
    for t in faces:
        tc = ((t[0] + t[2] / 2) / W_t, (t[1] + t[3] / 2) / H_t)
        best = min(anchor_faces, key=lambda a: np.hypot((a[0] + a[2] / 2) / W_a - tc[0], (a[1] + a[3] / 2) / H_a - tc[1]))
        dd = np.hypot((best[0] + best[2] / 2) / W_a - tc[0], (best[1] + best[3] / 2) / H_a - tc[1])
        if dd < tol: pairs.append((best, t))
    return pairs
