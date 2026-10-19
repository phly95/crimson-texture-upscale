"""zach_gypsy.py: warp the pack's HD '48gypsy' crystal-ball emblem (greyscale) onto the nose-art oval of the ZACHARYANDPLANE anchor crop -> face_audit/zach/gcomp.png"""
import sys, numpy as np, cv2
from PIL import Image
TH = float(sys.argv[1]) if len(sys.argv) > 1 else 38.0          # clockwise rotation of the emblem (its script then runs ~40-50 deg down to the right)
SZ = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
em = np.asarray(Image.open("face_audit/zach/gypsy_hd.png").convert("RGBA"), np.float32) / 255
ys, xs = np.where(em[..., 3] > 0.05); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
em = em[y0:y1 + 1, x0:x1 + 1]; h, w = em.shape[:2]; print("emblem", w, h)
ref = np.asarray(Image.open("face_audit/zach/ref.png").convert("RGB"), np.float32)
C = np.array([298.0, 448.0]); A, B, LONG = 335.0, 168.0, np.deg2rad(62.0)
c0 = np.array([w / 2, h / 2]); r = max(w, h) / 2; s = A * SZ / r
th = np.deg2rad(TH)
R1 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
n = np.array([np.cos(LONG + np.pi / 2), np.sin(LONG + np.pi / 2)])
Sq = np.eye(2) - (1 - B / A) * np.outer(n, n)             # squash across the ellipse's long axis
M = Sq @ R1 * s
t = C - M @ c0
Mat = np.hstack([M, t[:, None]])
g = em[..., :3].mean(-1, keepdims=True); emg = np.concatenate([np.repeat(g, 3, -1), em[..., 3:]], -1)
wp = cv2.warpAffine(emg, Mat, (1024, 1024), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4, borderValue=(0, 0, 0, 0))
a = np.clip(wp[..., 3:], 0, 1)
# soften old badge underneath: dark fill inside the ellipse so no skull shows through
m = np.zeros((1024, 1024), np.float32); cv2.ellipse(m, (int(C[0]), int(C[1])), (int(A * 1.0), int(B * 1.02)), 62, 0, 360, 1, -1)
m = cv2.GaussianBlur(m, (0, 0), 4)[..., None]
base = ref * (1 - m) + ref.mean(-1, keepdims=True) * 0.35 * m
out = base * (1 - a) + (wp[..., :3] * 255) * a
Image.fromarray(out.clip(0, 255).astype(np.uint8)).save("face_audit/zach/gcomp.png")
