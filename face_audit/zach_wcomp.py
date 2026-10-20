"""zach_wcomp.py: composite of the wrapped emblem layer on the anchor's original redraw window -> face_audit/zach/wcomp.png (1024, window (0,40,540,580) of the anchor)"""
import sys, numpy as np, cv2
from PIL import Image
Z = "/home/deck/workspace/csvm-hd/face_audit/zach/"
ref = np.asarray(Image.open(Z + "ref.png").convert("RGB"), np.float32)             # window of the anchor's own redraw, 1024
L = np.asarray(Image.open(Z + "wrap_layer.png").convert("RGBA"), np.float32) / 255
S = 1024 / 540
M = np.array([[S, 0, 0], [0, S, -40 * S]], np.float32)
Lw = cv2.warpAffine(L, M, (1024, 1024), flags=cv2.INTER_LANCZOS4)
a = np.clip(Lw[..., 3:], 0, 1)
fill = cv2.GaussianBlur(cv2.dilate((a[..., 0] > 0.3).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(np.float32), (0, 0), 8)[..., None]
base = ref * (1 - fill) + ref.mean(-1, keepdims=True) * 0.35 * fill
out = base * (1 - a) + Lw[..., :3] * 255 * a
Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(Z + "wcomp.png")
