"""headtex_skin.py: shift the warm skin pixels of one texture to the mean skin colour of another (shading kept)"""
import cv2, numpy as np
def _lab(x): return cv2.cvtColor(x, cv2.COLOR_RGB2LAB).astype(np.float32)
def skin(l):
    L, a, b = l[..., 0] * 100 / 255, l[..., 1] - 128, l[..., 2] - 128; return (a > 6) & (b > 14) & (a < 30) & (L > 20) & (L < 75)
def mean_skin(img, rows=(0.0, 1.0)):
    l = _lab(img); m = skin(l); m[:int(img.shape[0] * rows[0])] = False; m[int(img.shape[0] * rows[1]):] = False; return l[m].mean(0)
def match(img, target_mean):
    l = _lab(img); m = skin(l); w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2)[..., None]; out = l + w * (target_mean - l[m].mean(0))
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
