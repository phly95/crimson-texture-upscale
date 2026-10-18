"""recipes/common.py: Shared helpers and utilities for sprite recipes.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
from PIL import Image
import fx

def get_target_dims(stem):
    import json
    meta = json.loads(Path(f"sprites/{stem}/meta.json").read_text())
    return meta["target"], meta["orig_size"]

def smooth_envelope(orig, target_size):
    tw, th = target_size
    orig_up = cv2.resize(orig, (tw, th), interpolation=cv2.INTER_LANCZOS4)
    return np.clip(orig_up, 0, 1)

def polar_flow(W, H, cx=0.5, cy=0.5):
    """Return r, theta, tangent vector (tx, ty), radial vector (rx, ry)."""
    r = fx.radius(W, H, cx, cy)
    theta = fx.angle(W, H, cx, cy)
    tx = -np.sin(theta).astype(np.float32)
    ty = np.cos(theta).astype(np.float32)
    rx = np.cos(theta).astype(np.float32)
    ry = np.sin(theta).astype(np.float32)
    return r, theta, (tx, ty), (rx, ry)

def ridge_noise(W, H, scale=16, octaves=4, seed=0, gamma=2.0, tile=False):
    """Filamentary/crease noise 0..1."""
    n = fx.fbm(W, H, scale=scale, octaves=octaves, seed=seed, tile=tile)
    ridges = (1.0 - np.abs(n - 0.5) * 2.0) ** gamma
    return ridges.astype(np.float32)

def periodic_resize(img, target_size, interpolation=cv2.INTER_CUBIC):
    tw, th = target_size
    h, w = img.shape[:2]
    padded = np.pad(img, ((2, 2), (2, 2)) + (((0, 0),) if img.ndim == 3 else ()), mode="wrap")
    scale_x = tw / w
    scale_y = th / h
    pw = int(round((w + 4) * scale_x))
    ph = int(round((h + 4) * scale_y))
    up = cv2.resize(padded, (pw, ph), interpolation=interpolation)
    ox = int(round(2 * scale_x))
    oy = int(round(2 * scale_y))
    return up[oy:oy+th, ox:ox+tw]

def vtile_resize(img, target_size, interpolation=cv2.INTER_CUBIC):
    """Periodic wrapping vertically, reflection horizontally."""
    tw, th = target_size
    h, w = img.shape[:2]
    padded_v = np.pad(img, ((2, 2), (0, 0)) + (((0, 0),) if img.ndim == 3 else ()), mode="wrap")
    padded = np.pad(padded_v, ((0, 0), (2, 2)) + (((0, 0),) if img.ndim == 3 else ()), mode="reflect")
    scale_x = tw / w
    scale_y = th / h
    pw = int(round((w + 4) * scale_x))
    ph = int(round((h + 4) * scale_y))
    up = cv2.resize(padded, (pw, ph), interpolation=interpolation)
    ox = int(round(2 * scale_x))
    oy = int(round(2 * scale_y))
    return up[oy:oy+th, ox:ox+tw]

def htile_resize(img, target_size, interpolation=cv2.INTER_CUBIC):
    """Periodic wrapping horizontally, reflection vertically."""
    tw, th = target_size
    h, w = img.shape[:2]
    padded_h = np.pad(img, ((0, 0), (2, 2)) + (((0, 0),) if img.ndim == 3 else ()), mode="wrap")
    padded = np.pad(padded_h, ((2, 2), (0, 0)) + (((0, 0),) if img.ndim == 3 else ()), mode="reflect")
    scale_x = tw / w
    scale_y = th / h
    pw = int(round((w + 4) * scale_x))
    ph = int(round((h + 4) * scale_y))
    up = cv2.resize(padded, (pw, ph), interpolation=interpolation)
    ox = int(round(2 * scale_x))
    oy = int(round(2 * scale_y))
    return up[oy:oy+th, ox:ox+tw]

def calibrate_alpha(a_hd, a_orig, strict_zero=True, iterations=2, tile=False, vtile=False, htile=False):
    """Performs residual calibration so that Image.BOX downsampling matches a_orig."""
    H_hd, W_hd = a_hd.shape
    H_orig, W_orig = a_orig.shape

    a_cur = np.clip(a_hd.copy(), 0, 1)

    for _ in range(iterations):
        im_a = Image.fromarray((a_cur * 255.0 + 0.5).astype(np.uint8))
        a_down = np.asarray(im_a.resize((W_orig, H_orig), Image.BOX), np.float32) / 255.0

        diff = a_orig - a_down
        if tile:
            diff_up = periodic_resize(diff, (W_hd, H_hd), cv2.INTER_CUBIC)
            a_cur = np.clip(a_cur + diff_up, 0, 1)
        elif vtile:
            diff_up = vtile_resize(diff, (W_hd, H_hd), cv2.INTER_CUBIC)
            a_cur = np.clip(a_cur + diff_up, 0, 1)
        elif htile:
            diff_up = htile_resize(diff, (W_hd, H_hd), cv2.INTER_CUBIC)
            a_cur = np.clip(a_cur + diff_up, 0, 1)
        else:
            diff_up = cv2.resize(diff, (W_hd, H_hd), interpolation=cv2.INTER_CUBIC)
            support = fx.dilate((a_orig > 0).astype(np.float32), 1)
            support_up = cv2.resize(support, (W_hd, H_hd), interpolation=cv2.INTER_LINEAR)
            a_cur = np.clip(a_cur + diff_up * support_up, 0, 1)

    if strict_zero and not tile and not vtile and not htile:
        outer_zero = cv2.resize((fx.dilate((a_orig > 0).astype(np.float32), 1) == 0).astype(np.float32), (W_hd, H_hd), interpolation=cv2.INTER_NEAREST)
        a_cur = np.where(outer_zero > 0.5, 0.0, a_cur)

    return np.clip(a_cur, 0, 1)

def calibrate_color(rgb_hd, rgb_orig, a_orig, iterations=1, tile=False):
    """Subtle color residual calibration to ensure CIE dE stays minimal while keeping HD texture."""
    H_hd, W_hd, _ = rgb_hd.shape
    H_orig, W_orig, _ = rgb_orig.shape

    rgb_cur = np.clip(rgb_hd.copy(), 0, 1)
    for _ in range(iterations):
        im = Image.fromarray((rgb_cur * 255.0 + 0.5).astype(np.uint8))
        down = np.asarray(im.resize((W_orig, H_orig), Image.BOX), np.float32) / 255.0
        diff = rgb_orig - down
        if tile:
            diff_up = periodic_resize(diff, (W_hd, H_hd), cv2.INTER_CUBIC)
            rgb_cur = np.clip(rgb_cur + diff_up, 0, 1)
        else:
            diff_up = cv2.resize(diff, (W_hd, H_hd), interpolation=cv2.INTER_CUBIC)
            support = cv2.resize((a_orig > 0.01).astype(np.float32), (W_hd, H_hd), interpolation=cv2.INTER_LINEAR)
            rgb_cur = np.clip(rgb_cur + diff_up * support[..., None], 0, 1)
    return rgb_cur
