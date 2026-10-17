"""bloom_alpha.py: keep a soft glow/bloom (fire, explosion, flare, smoke) from the ORIGINAL when the redraw's key lost it.
alpha = max(original alpha (premultiplied-bicubic upscale), the redraw's soft-keyed alpha); colour = the redraw's flame colour where it has coverage, fading into the original's own glow colour where it has none."""
import numpy as np, cv2
from PIL import Image


def bloom(keyed_rgba, orig_rgba_path, gain=1.0, gmax=None):
    k = keyed_rgba.astype(np.float32) / 255.0
    H, W = k.shape[:2]
    o = np.asarray(Image.open(orig_rgba_path).convert('RGBA')).astype(np.float32) / 255.0
    pm = np.dstack([o[..., :3] * o[..., 3:4], o[..., 3:4]])
    up = cv2.resize(pm, (W, H), interpolation=cv2.INTER_CUBIC)
    ao = np.clip(up[..., 3:4] * gain, 0, 1)
    co = np.clip(up[..., :3] / np.maximum(up[..., 3:4], 1e-3), 0, 1)
    ak = k[..., 3:4]
    a = np.maximum(ao, ak)
    kr, kg, kb = k[..., 0:1], k[..., 1:2], k[..., 2:3]
    warm = np.clip((kr - kb) / 0.28, 0, 1)                      # flame edges are warm and bright; grey/olive haze and dark rims are not
    bright = np.clip((0.3 * kr + 0.59 * kg + 0.11 * kb) / 0.35, 0, 1)
    w_edge = np.clip(ak / 0.9, 0, 1) * warm * bright
    notgreen = np.ones_like(ak)
    if gmax is not None:                                       # screen-tinted halo (orange mixed with green, olive/yellow-green: green close to red and almost no blue); cream/white keep their blue
        olive = (kg / np.maximum(kr, 1e-3) > gmax) & (kb / np.maximum(kg, 1e-3) < 0.45)
        notgreen = cv2.GaussianBlur((~olive).astype(np.float32)[..., 0], (0, 0), 2)[..., None]
        w_edge = w_edge * notgreen
    r = max(3, int(round(0.016 * max(H, W))))
    deep = cv2.erode((ak[..., 0] > 0.9).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(np.float32)[..., None]
    w = np.maximum(w_edge, deep * notgreen)                                  # deep interior: always the redraw's colour; the rim band must also be warm and bright
    w = cv2.GaussianBlur(w, (0, 0), 3)[..., None] if w.ndim == 2 else cv2.GaussianBlur(w[..., 0], (0, 0), 3)[..., None]
    col = k[..., :3] * w + co * (1 - w)
    return (np.dstack([col, a]) * 255 + 0.5).clip(0, 255).astype(np.uint8)


def alpha_from_orig(keyed_rgba, orig_rgba_path, gain=1.3, core_only=False):
    """for light flares/glints: the ORIGINAL's smooth alpha (premultiplied-bicubic upscale, x gain) replaces the redraw's alpha, so a hard-edged disc or too-wide spikes in the redraw
    become the original's soft falloff; colour = the redraw's (soft-keyed) colour where it has coverage, else the original's own."""
    k = keyed_rgba.astype(np.float32) / 255.0
    H, W = k.shape[:2]
    o = np.asarray(Image.open(orig_rgba_path).convert('RGBA')).astype(np.float32) / 255.0
    pm = np.dstack([o[..., :3] * o[..., 3:4], o[..., 3:4]])
    up = cv2.resize(pm, (W, H), interpolation=cv2.INTER_CUBIC)
    ao = np.clip(up[..., 3:4] * gain, 0, 1)
    co = np.clip(up[..., :3] / np.maximum(up[..., 3:4], 1e-3), 0, 1)
    w = np.clip(k[..., 3:4] / 0.5, 0, 1)
    if core_only:                                              # keep the redraw's colour only deep inside; the glow band uses the original's colour (the redraw's glow is olive-tinted)
        r = max(3, int(round(0.04 * max(H, W))))
        deep = cv2.erode((k[..., 3] > 0.9).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(np.float32)
        w = cv2.GaussianBlur(deep, (0, 0), r / 2.0)[..., None]
    col = k[..., :3] * w + co * (1 - w)
    return (np.dstack([col, ao]) * 255 + 0.5).clip(0, 255).astype(np.uint8)
