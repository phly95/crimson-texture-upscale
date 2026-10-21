"""outline_fit.py: UI pictures whose silhouette other art is drawn against (the memento polaroids and the stack of sheets under them) keep the original's outline.

A redraw's polaroid is a few px off the original's corners, so the sheets baked into momento_shad (and the shadow the script centres under the memento) stop lining up and the
background shows through along the edges. polaroid_fit() warps the redraw's four corners onto the original's and takes the alpha from the original; shadow_from_orig() makes the
sheet stack by a smooth upscale of the original, since a redraw cannot be made to agree with a neighbouring picture's edges.
"""
import cv2
import numpy as np


def quad(a):
    """corners (TL, TR, BR, BL) of the largest opaque shape in an alpha channel"""
    m = (a > 128).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    m = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    h = cv2.convexHull(c)
    per = cv2.arcLength(h, True)
    ap = h
    for eps in np.linspace(0.005, 0.08, 40):
        ap = cv2.approxPolyDP(h, eps * per, True)
        if len(ap) <= 4:
            break
    p = ap[:, 0, :].astype(np.float32) if len(ap) == 4 else cv2.boxPoints(cv2.minAreaRect(c))
    s = p.sum(1); d = p[:, 0] - p[:, 1]
    return np.array([p[np.argmin(s)], p[np.argmax(d)], p[np.argmax(s)], p[np.argmin(d)]], np.float32)


def fill_outside(rgb, solid):
    """colour of every pixel outside `solid` := its nearest solid pixel (nothing from beyond the picture's edge survives)"""
    hole = ~solid
    if not hole.any() or hole.all():
        return rgb
    _, lab = cv2.distanceTransformWithLabels(hole.astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(lab.max() + 1, np.int64)
    ys, xs = np.nonzero(solid)
    lut[lab[solid]] = np.arange(len(ys))
    idx = lut[lab[hole]]
    out = rgb.copy()
    out[hole] = rgb[ys[idx], xs[idx]]
    return out


def smooth_alpha(a8, size, sigma=0.7):
    """the original's alpha at `size`: blurred a little so its 1 px stair-steps do not become 4 px ones, edge re-sharpened over ~2 px"""
    a = cv2.GaussianBlur(a8.astype(np.float32) / 255, (0, 0), sigma)
    t = np.clip((cv2.resize(a, size, interpolation=cv2.INTER_CUBIC) - 0.25) / 0.5, 0, 1)
    return t * t * (3 - 2 * t)


def polaroid_fit(hd, o):
    """hd, o = RGBA uint8 arrays (the redraw, the original at its own size) -> RGBA at the redraw's size with the original's outline"""
    H, W = hd.shape[:2]
    sc = W / o.shape[1]
    M = cv2.getPerspectiveTransform(quad(hd[..., 3]), quad(o[..., 3]) * sc)
    w = cv2.warpPerspective(hd.astype(np.float32), M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    solid = cv2.erode((w[..., 3] >= 250).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3) > 0
    rgb = fill_outside(np.clip(w[..., :3], 0, 255), solid)
    return np.dstack([rgb, smooth_alpha(o[..., 3], (W, H)) * 255]).round().astype(np.uint8)


def shadow_from_orig(o, size):
    """smooth upscale of the original RGBA (premultiplied, so the shadow keeps its colour) with a light unsharp on alpha"""
    f = o.astype(np.float32) / 255
    P = np.clip(cv2.resize(f[..., :3] * f[..., 3:], size, interpolation=cv2.INTER_CUBIC), 0, None)
    A = np.clip(cv2.resize(f[..., 3], size, interpolation=cv2.INTER_CUBIC), 0, 1)
    A2 = np.clip(A + 0.6 * (A - cv2.GaussianBlur(A, (0, 0), 2.0)), 0, 1)
    rgb = np.where(A[..., None] > 1e-3, P / np.maximum(A[..., None], 1e-3), 0).clip(0, 1)
    return np.dstack([rgb * 255, A2 * 255]).round().astype(np.uint8)
