"""headtex_feat.py: feature masks (goggles, chin strap, skin) of a front head view inside the face box + hand traces (nose, mouth); IoU between a redraw and the reference; flow alignment on the feature field"""
import sys, json
from pathlib import Path
import cv2, numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "face_audit/headtex"
BOX = (340, 330, 660, 700)                                      # x0, y0, x1, y1 of the face in the 1024 front render
NOSE = np.array([(512, 430), (503, 445), (496, 465), (494, 490), (498, 510), (494, 522), (496, 534), (512, 541), (528, 534), (530, 522), (526, 510), (530, 490), (528, 465), (521, 445)], np.int32)   # hand trace, centred on the measured nose axis x=512
MOUTH = np.array([(452, 564), (561, 566)], np.int32)             # the user's measured line
def masks(rgb):
    x0, y0, x1, y1 = BOX; lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32); L, a, b = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
    sub = np.zeros(L.shape, bool); sub[y0:y1, x0:x1] = True
    gog = sub & (L < 42) & (np.arange(L.shape[0])[:, None] < 500); strap = sub & (L < 38) & (np.arange(L.shape[0])[:, None] > 598)
    skin = sub & (L > 42) & (a > 3) & (b > 6) & ~gog & ~strap
    cl = lambda m, k: cv2.morphologyEx(cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)), cv2.MORPH_CLOSE, np.ones((2 * k, 2 * k), np.uint8)) > 0
    gb = np.zeros(L.shape, bool); gb[342:490, 372:652] = True; ns = gb & ~((L > 42) & (a > 3) & (b > 6))                      # goggle object = everything in the goggle box that is not skin (dark padding + lens + bright steel rim)
    return dict(goggles=cl(ns, 7), goggles_dark=cl(gog, 7), strap=cl(strap, 7), skin=cl(skin, 5))
def iou(a, b): return float((a & b).sum() / max(1, (a | b).sum()))
def compare(ref, cand): mr, mc = masks(ref), masks(cand); return {k: iou(mr[k], mc[k]) for k in ("goggles", "strap", "skin")}
def field(rgb, hand=False):
    m = masks(rgb); f = sum(cv2.GaussianBlur(m[k].astype(np.float32), (0, 0), 5) * w for k, w in (("goggles", 1.0), ("strap", 1.0), ("skin", 0.6)))
    return f
if __name__ == "__main__":
    ref = np.array(Image.open(D / "v_cphead_front.png").convert("RGB"))
    for f in sys.argv[1:]:
        c = np.array(Image.open(D / f).convert("RGB")); print(f, {k: round(v, 3) for k, v in compare(ref, c).items()})

def align_field(ref, cand, smooth=10.0, clip=26.0):
    """warp cand so that its goggle / strap / skin outlines land on the reference's (DIS flow between the blurred feature fields), only inside the face box"""
    fr, fc = field(ref), field(cand); u8 = lambda f: np.clip(f / max(f.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); fl = dis.calc(u8(fr), u8(fc), None)
    x0, y0, x1, y1 = BOX; w = np.zeros(fr.shape, np.float32); w[y0:y1, x0:x1] = 1; w = cv2.GaussianBlur(w, (0, 0), 14)
    fl = cv2.GaussianBlur(fl, (0, 0), smooth) * w[..., None]; m = np.hypot(fl[..., 0], fl[..., 1]); fl *= np.minimum(1, clip / np.maximum(m, 1e-6))[..., None]
    h, ww = ref.shape[:2]; gx, gy = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(cand, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT), float(m.max())


def nose_axis(rgb, y0=440, y1=548, half=45, search=24):
    """where is the nose's own mirror axis (x) and how symmetric is the nose about it: best offset d in [-search, search] of corr(crop, mirrored crop) on the nose rows"""
    g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 2.5); best = (-9, 0)
    for d in range(-search, search + 1):
        c = g[y0:y1, 512 + d - half:512 + d + half]; f = c[:, ::-1]; a, b = c - c.mean(), f - f.mean(); r = float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum() + 1e-9))
        if r > best[0]: best = (r, d)
    return 512 + best[1], best[0]

def outline_metrics(crop_rgb, ref_mask, harm=5):
    """goggle outline of a (1008x480) goggle crop: IoU with the vector shape and wobble = RMS distance (crop px) between the outline of the left lens group and its Fourier-smoothed version"""
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32); L = lab[..., 0] * 100 / 255
    m = (L < 42).astype(np.uint8); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((31, 31), np.uint8)); m[:, 500:] = 0
    r = ref_mask.copy(); r[:, 500:] = 0
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs: return dict(iou=0.0, wobble=99.0)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float64); d = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]; t = np.linspace(0, d[-1], 720, endpoint=False)
    p = np.c_[np.interp(t, d, c[:, 0]), np.interp(t, d, c[:, 1])]; Z = np.fft.fft(p[:, 0] + 1j * p[:, 1]); k = np.zeros_like(Z); k[:harm + 1] = 1; k[-harm:] = 1; zs = np.fft.ifft(Z * k)
    return dict(iou=iou(m > 0, r > 0), wobble=float(np.sqrt(np.mean(np.abs(p[:, 0] + 1j * p[:, 1] - zs) ** 2))))
