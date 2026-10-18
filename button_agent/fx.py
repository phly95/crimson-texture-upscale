"""fx.py: a small layer toolkit for drawing soft effect sprites in code (numpy + OpenCV + skia).

A LAYER is a float32 array (H, W, 4): straight (non-premultiplied) RGB in 0..1 and alpha in 0..1. Everything returns new arrays.
Typical recipe:
    import fx; H = W = 512
    n = fx.fbm(W, H, scale=40, octaves=5, seed=3, tile=True)                 # 0..1 noise
    ring = fx.smoothstep(0.30, 0.34, fx.radius(W, H)) * (1 - fx.smoothstep(0.38, 0.46, fx.radius(W, H)))
    a = fx.levels(ring * (0.6 + 0.8 * n), 0.0, 1.0) * 0.55                    # alpha
    layer = fx.solid(W, H, (0.8, 0.85, 0.9), a)                               # colour + alpha
    out = fx.blend(fx.empty(W, H), layer, "normal")
    fx.save(out, "draw.png")
Main pieces: coordinates (grid, radius, angle), noise (fbm, warp_noise), shaping (smoothstep, levels, curve), colour (ramp, solid, from_gray),
compositing (blend with normal/add/screen/multiply/overlay, mask, over_bg), filters (blur, glow, dilate), vector painting with skia (paint, strokes_flow),
and analysis of the ORIGINAL (load, radial_profile, alpha_stats, palette) so numbers come from it instead of guesses.
"""
import numpy as np
import cv2
from PIL import Image

# ---- io -------------------------------------------------------------------------------------------------------------------------------
def load(path, size=None):
    """RGBA layer from a PNG (straight alpha, 0..1); size=(W, H) resizes with Lanczos (colour premultiplied so edges don't halo)."""
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im, np.float32) / 255
    if size and tuple(size) != im.size:
        pm = np.dstack([a[..., :3] * a[..., 3:], a[..., 3:]])
        pm = cv2.resize(pm, tuple(size), interpolation=cv2.INTER_LANCZOS4)
        al = np.clip(pm[..., 3:], 0, 1)
        a = np.dstack([np.clip(pm[..., :3] / np.maximum(al, 1e-4), 0, 1), al])
    return a

def save(layer, path):
    """write a layer as PNG; colour under alpha 0 is set to black (never a key colour)."""
    l = np.clip(layer, 0, 1).copy()
    l[..., :3] *= (l[..., 3:] > 0)
    Image.fromarray((l * 255 + .5).astype(np.uint8), "RGBA").save(path)

# ---- coordinates ----------------------------------------------------------------------------------------------------------------------
def grid(W, H):
    """x, y in 0..1 (pixel centres)"""
    return np.meshgrid((np.arange(W) + .5) / W, (np.arange(H) + .5) / H)

def radius(W, H, cx=.5, cy=.5):
    """distance from (cx, cy) in image units: 0.5 reaches the edge midpoint of a square image"""
    x, y = grid(W, H)
    return np.hypot((x - cx) * W / max(W, H), (y - cy) * H / max(W, H))

def angle(W, H, cx=.5, cy=.5):
    x, y = grid(W, H)
    return np.arctan2(y - cy, x - cx)

# ---- noise ----------------------------------------------------------------------------------------------------------------------------
def _field(W, H, cell, rng, tile):
    gw, gh = max(2, int(round(W / cell))), max(2, int(round(H / cell)))
    g = rng.random((gh, gw)).astype(np.float32)
    if tile:                                                         # periodic cubic interpolation of a wrapped lattice
        g = np.pad(g, 2, mode="wrap")
        up = cv2.resize(g, (int((gw + 4) * cell), int((gh + 4) * cell)), interpolation=cv2.INTER_CUBIC)
        o = int(2 * cell)
        up = up[o:o + int(gh * cell), o:o + int(gw * cell)]
        return cv2.resize(up, (W, H), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)

def fbm(W, H, scale=32, octaves=4, seed=0, gain=.5, tile=False):
    """fractal value noise 0..1. scale = size in pixels of the biggest feature; tile=True makes it wrap on both axes."""
    rng = np.random.default_rng(seed)
    out, amp, tot, cell = np.zeros((H, W), np.float32), 1., 0., float(scale)
    for _ in range(octaves):
        out += amp * _field(W, H, max(cell, 2), rng, tile)
        tot += amp; amp *= gain; cell /= 2
    out /= tot
    lo, hi = np.percentile(out, [1, 99])
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)

def warp(img, dx, dy):
    """displace img (any channels) by dx, dy pixels (arrays HxW); edges wrap (good for tiles)"""
    H, W = dx.shape
    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(img.astype(np.float32), x + dx.astype(np.float32), y + dy.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

def warp_noise(img, W, H, amount=20, scale=60, seed=0, tile=False):
    """domain warp: the cheap way to turn clean shapes into organic ones"""
    dx = (fbm(W, H, scale, 3, seed, tile=tile) - .5) * 2 * amount
    dy = (fbm(W, H, scale, 3, seed + 101, tile=tile) - .5) * 2 * amount
    return warp(img, dx, dy)

# ---- shaping --------------------------------------------------------------------------------------------------------------------------
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)

def levels(x, lo=0., hi=1., gamma=1.):
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1) ** gamma

def curve(x, pts):
    """piecewise-linear remap through [(in, out), ...]"""
    p = np.array(pts, np.float32)
    return np.interp(x, p[:, 0], p[:, 1]).astype(np.float32)

# ---- colour ---------------------------------------------------------------------------------------------------------------------------
def ramp(v, stops):
    """colour ramp: v in 0..1 -> RGB through [(pos, (r, g, b)), ...] (values 0..1); the 'gradient map' of Photoshop"""
    pos = [s[0] for s in stops]
    return np.dstack([np.interp(v, pos, [s[1][c] for s in stops]) for c in range(3)]).astype(np.float32)

def solid(W, H, rgb, alpha=1.):
    out = np.zeros((H, W, 4), np.float32)
    out[..., :3] = rgb
    out[..., 3] = alpha
    return out

def empty(W, H):
    return np.zeros((H, W, 4), np.float32)

def from_gray(v, ramp_stops=None, alpha=None):
    """grey 0..1 -> layer; with ramp_stops it is a gradient map; alpha defaults to v"""
    rgb = ramp(v, ramp_stops) if ramp_stops else np.dstack([v, v, v])
    return np.dstack([rgb, v if alpha is None else alpha]).astype(np.float32)

# ---- compositing ----------------------------------------------------------------------------------------------------------------------
def _blend_rgb(b, s, mode):
    if mode == "add":      return np.minimum(b + s, 1)
    if mode == "screen":   return 1 - (1 - b) * (1 - s)
    if mode == "multiply": return b * s
    if mode == "overlay":  return np.where(b < .5, 2 * b * s, 1 - 2 * (1 - b) * (1 - s))
    return s

def blend(base, layer, mode="normal", opacity=1., mask=None):
    """composite layer over base (both straight-alpha layers). modes: normal add screen multiply overlay. mask: HxW 0..1 multiplies layer alpha."""
    sa = layer[..., 3] * opacity * (1 if mask is None else mask)
    ba = base[..., 3]
    oa = sa + ba * (1 - sa)
    mixed = (1 - ba[..., None]) * layer[..., :3] + ba[..., None] * _blend_rgb(base[..., :3], layer[..., :3], mode)   # blend only where base exists
    num = base[..., :3] * (ba * (1 - sa))[..., None] + mixed * sa[..., None]
    out = np.zeros_like(base)
    out[..., :3] = num / np.maximum(oa, 1e-6)[..., None]
    out[..., 3] = oa
    return np.clip(out, 0, 1)

def over_bg(layer, bg=(.5, .5, .5)):
    """flatten on a background colour: how the engine will see it before lighting"""
    return layer[..., :3] * layer[..., 3:] + np.array(bg, np.float32) * (1 - layer[..., 3:])

# ---- filters --------------------------------------------------------------------------------------------------------------------------
def blur(x, sigma, wrap=False):
    """gaussian blur of an HxW mask or HxWxC layer. For a layer, colour is blurred premultiplied."""
    mode = cv2.BORDER_WRAP if wrap else cv2.BORDER_REFLECT
    if x.ndim == 2:
        return cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigma, borderType=mode)
    pm = np.dstack([x[..., :3] * x[..., 3:], x[..., 3:]])
    pm = cv2.GaussianBlur(pm, (0, 0), sigma, borderType=mode)
    a = pm[..., 3:]
    return np.dstack([pm[..., :3] / np.maximum(a, 1e-6), a])

def glow(layer, sigma, strength=1.):
    """additive bloom of a layer's own light; returns a layer to blend 'add' or 'screen'"""
    g = blur(layer, sigma)
    g[..., 3] = np.clip(g[..., 3] * strength, 0, 1)
    return g

def dilate(mask, px):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.float32), k)

# ---- vector painting with skia (brush strokes, soft gradients, real anti-aliasing) -----------------------------------------------------
def paint(W, H, draw):
    """draw(canvas, skia) paints on a transparent skia canvas; returns a straight-alpha layer. Example:
        def d(c, sk):
            p = sk.Paint(AntiAlias=True, Shader=sk.GradientShader.MakeRadial((W/2, H/2), W/3, [sk.ColorSetARGB(255,255,200,80), sk.ColorSetARGB(0,255,80,0)], [0, 1]))
            c.drawCircle(W/2, H/2, W/3, p)
        layer = fx.paint(W, H, d)"""
    import skia
    surf = skia.Surface(W, H)
    c = surf.getCanvas()
    c.clear(skia.ColorTRANSPARENT)
    draw(c, skia)
    img = surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kUnpremul_AlphaType)
    return np.asarray(img, np.float32) / 255

def strokes_flow(W, H, field_angle, n=400, length=60, width=(1, 3), color=(1, 1, 1), alpha=(.2, .6), seed=0, blur_sigma=0):
    """thin wispy strokes that follow a flow field (field_angle: HxW radians). Gives spray, wake streaks, flame tongues. Returns a layer."""
    rng = np.random.default_rng(seed)
    def d(c, sk):
        for _ in range(n):
            x, y = rng.random() * W, rng.random() * H
            path = sk.Path(); path.moveTo(x, y)
            step = length / 12
            for _ in range(12):
                a = field_angle[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))]
                x += np.cos(a) * step; y += np.sin(a) * step
                path.lineTo(x, y)
            p = sk.Paint(AntiAlias=True, Style=sk.Paint.kStroke_Style, StrokeWidth=float(rng.uniform(*width)), StrokeCap=sk.Paint.kRound_Cap,
                         Color4f=sk.Color4f(*color, float(rng.uniform(*alpha))))
            c.drawPath(path, p)
    out = paint(W, H, d)
    return blur(out, blur_sigma) if blur_sigma else out

# ---- analysis of the original ---------------------------------------------------------------------------------------------------------
def radial_profile(alpha, bins=24):
    """mean alpha at each radius from the image centre (0..0.5*sqrt2): what shape does the original's falloff have?"""
    H, W = alpha.shape
    r = radius(W, H)
    idx = np.minimum((r / r.max() * bins).astype(int), bins - 1)
    return np.array([alpha[idx == i].mean() if (idx == i).any() else 0 for i in range(bins)])

def alpha_stats(layer):
    a = layer[..., 3]
    ys, xs = np.where(a > .5)
    bb = None if not len(ys) else (xs.min() / a.shape[1], ys.min() / a.shape[0], (xs.max() + 1) / a.shape[1], (ys.max() + 1) / a.shape[0])
    return {"coverage": float(a.mean()), "bbox_frac": bb, "soft_share": float(((a > .08) & (a < .92)).mean()), "max": float(a.max())}

def palette(layer, k=5):
    """k dominant colours (weighted by alpha) as 0..1 RGB, most common first"""
    a = layer[..., 3].reshape(-1)
    px = layer[..., :3].reshape(-1, 3)[a > .05].astype(np.float32)
    if len(px) < k:
        return [tuple(p) for p in px]
    _, lab, cen = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, .5), 3, cv2.KMEANS_PP_CENTERS)
    order = np.argsort(-np.bincount(lab.ravel(), minlength=k))
    return [tuple(float(v) for v in cen[i]) for i in order]
