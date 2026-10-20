"""headtex_snap.py: thin-plate warp of a redraw so its YuNet landmarks land on the reference's (face only: anchors on a ring around the face and the image corners stay put)"""
import sys
from pathlib import Path
import cv2, numpy as np
from scipy.interpolate import RBFInterpolator
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_score as S
def snap(ref, cand, ring=1.7, n=20):
    fr, fc = S.face(ref), S.face(cand)
    if fr is None or fc is None: return cand, None
    h, w = ref.shape[:2]; cx, cy = fr["box"][0] + fr["box"][2] / 2, fr["box"][1] + fr["box"][3] / 2; R = ring * max(fr["box"][2], fr["box"][3]) / 2
    a = np.linspace(0, 2 * np.pi, n, endpoint=False); anchors = np.c_[cx + R * np.cos(a), cy + R * np.sin(a)]
    anchors = np.vstack([anchors, [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]]])
    dst = np.vstack([fr["lm"], anchors]); src = np.vstack([fc["lm"], anchors])               # ref position -> where to read in the candidate
    k = RBFInterpolator(dst, src - dst, kernel="thin_plate_spline", smoothing=0.0)
    g = 8; xs, ys = np.meshgrid(np.arange(0, w + g, g), np.arange(0, h + g, g)); pts = np.c_[xs.ravel(), ys.ravel()].astype(float)
    off = k(pts).reshape(xs.shape + (2,)); off = cv2.resize(off.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(cand, gx + off[..., 0], gy + off[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT), float(np.abs(off).max())
