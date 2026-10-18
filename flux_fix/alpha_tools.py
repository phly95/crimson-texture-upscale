"""alpha_tools.py: helpers for the FLUX alpha fix (guide images, hybrid alpha imposition, review sheets).  Import from flux_fix/*.py run with `-I` and sys.path set.

  guide(stem, mode='hard'|'soft'|'mix', ...)  -> PIL RGB in in.png's size: original colours (premultiplied upscale, bled outward) on pure green (0,255,0)
  hybrid(gen_rgba_png, stem, out_png, ...)    -> generation's colours with the original alpha imposed (resized, smooth) and colour bled outward
  sheet(png, out_png)                          -> the picture on checker / dark / light / green side by side (for the eye)
"""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
G = np.array([0, 255, 0], np.float32)


def unit_of(stem):
    cat = json.load(open(ROOT / "catalog.json"))
    for uid, u in cat["units"].items():
        if u["stem"] == stem and not u["is_mip"]:
            return uid
    raise SystemExit(f"unknown stem {stem}")


def udir(stem):
    return ROOT / "pipeline" / "units" / unit_of(stem)


def _bleed(rgb, valid, iters=None):
    """fill rgb where ~valid by smooth outward extension (normalised blur at growing radii: no nearest-neighbour streaks)"""
    valid = valid.astype(bool)
    if valid.all() or not valid.any():
        return rgb
    w0 = valid.astype(np.float32)
    num0 = rgb * w0[..., None]
    out = rgb.copy()
    filled = valid.copy()
    sig = 1.5
    H, W = valid.shape
    while not filled.all() and sig < max(H, W):
        bw = cv2.GaussianBlur(w0, (0, 0), sig)
        bn = cv2.GaussianBlur(num0, (0, 0), sig)
        ok = (bw > 0.02) & ~filled
        out[ok] = bn[ok] / bw[ok][:, None]
        filled |= ok
        sig *= 1.7
    out[valid] = rgb[valid]
    return out


def orig_up(stem, size, kind=cv2.INTER_CUBIC):
    """(rgb float 0..255 with transparent colours bled outward, alpha float 0..1) of orig.png at `size` (W,H); premultiplied resize"""
    o = np.asarray(Image.open(udir(stem) / "orig.png").convert("RGBA"), np.float32)
    a = o[..., 3:] / 255
    pm = o[..., :3] * a
    W, H = size
    pmu = cv2.resize(pm, (W, H), interpolation=kind)
    au = np.clip(cv2.resize(a[..., 0], (W, H), interpolation=kind), 0, 1)
    rgb = np.clip(pmu / np.maximum(au[..., None], 1e-3), 0, 255)
    rgb = _bleed(rgb, au > 0.6)
    return rgb, au


def hard_mask(au, thr=0.5, smooth=2.0, aa=1.2):
    """smooth then threshold the upscaled alpha -> crisp silhouette (float 0..1, only ~1px anti-aliased)"""
    s = cv2.GaussianBlur(au, (0, 0), smooth) if smooth else au
    m = (s > thr).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), aa) if aa else m


def guide(stem, mode="hard", thr=0.5, smooth=2.0, aa=1.2, soft_gain=1.0, size=None, bg=None):
    """mode hard: crisp silhouette; soft: the (smooth) original alpha as blend with green; mix: hard silhouette, but zones where the original is
    clearly translucent (0.15<a<0.85 over a big area) keep their blend.  Colours inside from the original."""
    W, H = size or Image.open(udir(stem) / "in.png").size
    rgb, au = orig_up(stem, (W, H))
    hm = hard_mask(au, thr, smooth, aa)
    if mode == "fill":
        return Image.fromarray(np.clip(rgb + .5, 0, 255).astype(np.uint8))
    if mode == "hard":
        m = hm
    elif mode == "soft":
        m = np.clip(cv2.GaussianBlur(au, (0, 0), smooth) * soft_gain, 0, 1)
    else:
        sm = cv2.GaussianBlur(au, (0, 0), smooth)
        part = ((sm > 0.12) & (sm < 0.88)).astype(np.uint8)
        part = cv2.morphologyEx(part, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)).astype(np.float32)
        part = cv2.GaussianBlur(part, (0, 0), 4)
        m = hm * (1 - part) + sm * part
    out = rgb * m[..., None] + (G if bg is None else np.array(bg, np.float32)) * (1 - m[..., None])
    return Image.fromarray(np.clip(out + .5, 0, 255).astype(np.uint8))


def hybrid(gen_png, stem, out_png, smooth=1.0, thr=None, soft=True, expand=0, bleed=True):
    """take the generation's colours (RGBA from the keyed redraw; transparent colours bled outward) and impose the ORIGINAL alpha resized to the generation's size.
    soft=True keeps the original's soft values (smoothed bicubic), soft=False thresholds them for a crisp edge."""
    g = np.asarray(Image.open(gen_png).convert("RGBA"), np.float32)
    H, W = g.shape[:2]
    ga = g[..., 3] / 255
    rgb = _bleed(g[..., :3], ga > 0.5)
    _, au = orig_up(stem, (W, H))
    if smooth:
        au = cv2.GaussianBlur(au, (0, 0), smooth)
    if thr is not None:
        au = hard_mask(au, thr, 0, 1.0)
    if expand:
        au = cv2.dilate(au, np.ones((expand * 2 + 1,) * 2, np.uint8)) if expand > 0 else cv2.erode(au, np.ones((-expand * 2 + 1,) * 2, np.uint8))
    # make sure colours under every visible pixel are real picture colours, not green screen: bleed from the redraw's solid interior
    solid = (ga > 0.9) & (au > 0.5)
    solid = cv2.erode(solid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if bleed:
        rgb = _bleed(rgb, solid)
    else:                                    # full-frame redraw (no green screen): its own colours are valid everywhere
        rgb = g[..., :3]
    out = np.dstack([np.clip(rgb, 0, 255), np.clip(au, 0, 1) * 255]).astype(np.uint8)
    Image.fromarray(out, "RGBA").save(out_png)
    return out_png


def sheet(png, out, width=480):
    im = Image.open(png).convert("RGBA")
    W = width
    H = max(8, int(width * im.height / im.width))
    im = im.resize((W, H), Image.LANCZOS if im.width > W else Image.NEAREST)
    tiles = []
    yy, xx = np.mgrid[0:H, 0:W]
    g = (((xx // 16) + (yy // 16)) % 2 * 60 + 110).astype(np.uint8)
    for bg in ("chk", (15, 15, 18), (235, 235, 230), (0, 255, 0)):
        b = Image.fromarray(np.dstack([g, g, g])).convert("RGBA") if bg == "chk" else Image.new("RGBA", im.size, bg + (255,))
        b.alpha_composite(im)
        tiles.append(b.convert("RGB"))
    cols = 2 if H > 300 and W * 4 > 1900 else 2
    sh = Image.new("RGB", (W * 2 + 10, H * 2 + 10), (60, 60, 60))
    for i, t in enumerate(tiles):
        sh.paste(t, ((i % 2) * (W + 10), (i // 2) * (H + 10)))
    sh.save(out)
    return out


def compare(stem, cand, out):
    """alpha_tol's contact sheet via its CLI is fine; this just returns metrics"""
    import sys
    sys.path.insert(0, str(HERE))
    import tol
    return tol.metrics(udir(stem) / "orig.png", cand)


def multi(stem, pngs, out, width=420):
    """row per candidate: original | candidate on dark | on light | on green, with metrics in the file name order printed to stdout"""
    import sys
    sys.path.insert(0, str(HERE))
    import tol
    o = Image.open(udir(stem) / "orig.png").convert("RGBA")
    rows = []
    for p in pngs:
        im = Image.open(p).convert("RGBA")
        W = width; H = max(8, int(width * im.height / im.width))
        tiles = []
        oo = o.resize((W, H), Image.NEAREST if o.width * 4 <= W else Image.LANCZOS)
        yy, xx = np.mgrid[0:H, 0:W]
        g = (((xx // 16) + (yy // 16)) % 2 * 60 + 110).astype(np.uint8)
        b = Image.fromarray(np.dstack([g, g, g])).convert("RGBA"); b.alpha_composite(oo); tiles.append(b.convert("RGB"))
        imr = im.resize((W, H), Image.LANCZOS)
        for bg in ((15, 15, 18), (235, 235, 230), (0, 255, 0)):
            b = Image.new("RGBA", imr.size, bg + (255,)); b.alpha_composite(imr); tiles.append(b.convert("RGB"))
        rows.append(tiles)
        print(Path(p).parent.name, Path(p).name, tol.metrics(udir(stem) / "orig.png", p))
    H = rows[0][0].height; W = rows[0][0].width
    sh = Image.new("RGB", (W * 4 + 30, (H + 8) * len(rows)), (60, 60, 60))
    for r, tiles in enumerate(rows):
        for c, t in enumerate(tiles):
            sh.paste(t, (c * (W + 10), r * (H + 8)))
    sh.save(out)


def repair_solid(gen_png, stem, out_png, thr=0.95, erode=2):
    """where the ORIGINAL is solidly opaque (eroded a bit) but the generation has partial alpha / green leak: make it opaque with colours bled from the generation's solid pixels"""
    g = np.asarray(Image.open(gen_png).convert("RGBA"), np.float32)
    H, W = g.shape[:2]
    ga = g[..., 3] / 255
    _, au = orig_up(stem, (W, H))
    solid_o = (cv2.GaussianBlur(au, (0, 0), 1.5) > thr).astype(np.uint8)
    if erode:
        solid_o = cv2.erode(solid_o, np.ones((erode * 2 + 1,) * 2, np.uint8))
    bad = (solid_o > 0) & (ga < 0.98)
    good = ga > 0.98
    rgb = g[..., :3].copy()
    filled = _bleed(rgb, good & ~cv2.dilate(bad.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool))
    rgb[bad] = filled[bad]
    a = np.where(bad, 1.0, ga)
    Image.fromarray(np.dstack([np.clip(rgb, 0, 255), a * 255]).astype(np.uint8), "RGBA").save(out_png)
    return int(bad.sum())


def modulate(gen_png, stem, out_png, sigma=8, floor=0.0):
    """keep the generation's crisp SHAPE (its own alpha) but give it the original's local translucency level: alpha = gen_alpha * m, where m is the original's
    alpha averaged (normalised blur, sigma px at gen size) over the original's own cloth, extended outward smoothly.  For cloth/glass whose original alpha is ~0.7 everywhere."""
    g = np.asarray(Image.open(gen_png).convert("RGBA"), np.float32)
    H, W = g.shape[:2]
    ga = g[..., 3] / 255
    _, au = orig_up(stem, (W, H))
    w = (au > 0.15).astype(np.float32)
    num = cv2.GaussianBlur(au * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    m = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0)
    m = _bleed(np.dstack([m] * 3), den > 0.05)[..., 0]
    m = np.clip(m, floor, 1)
    out = np.dstack([g[..., :3], ga * m * 255]).astype(np.uint8)
    Image.fromarray(out, "RGBA").save(out_png)


def neon_colors(png_in, png_out, hi=170, tone=215, sigma=3, grow=0, keep=0.45):
    """for glow/neon art generated on a dark backing: dark pixels get the local neon hue at full brightness (so partial-alpha glow shows pink, not dark, on any backdrop)"""
    g = np.asarray(Image.open(png_in).convert("RGB"), np.float32)
    v = g.max(-1)
    b = cv2.GaussianBlur(g, (0, 0), sigma)
    # hue from the brighter surroundings: weight by brightness
    w = cv2.GaussianBlur(v[..., None] if False else v, (0, 0), sigma)
    bw = cv2.GaussianBlur(g * v[..., None], (0, 0), sigma) / np.maximum(cv2.GaussianBlur(v, (0, 0), sigma)[..., None], 1)
    hue = bw / np.maximum(bw.max(-1, keepdims=True), 1)
    t = np.clip(v / hi, 0, 1)[..., None]
    out = g * t + hue * tone * (1 - t)
    if grow:
        out = cv2.GaussianBlur(cv2.dilate(out, np.ones((grow, grow), np.uint8)), (0, 0), grow / 4)
        sat = (g.max(-1) - g.min(-1)) / np.maximum(g.max(-1), 1)
        sel = (v > 90) & (sat > 0.2)
        gh = np.median(g[sel] / g[sel].max(-1, keepdims=True), 0) if sel.any() else np.ones(3)
        out = out * keep + gh * tone * (1 - keep)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(png_out)


def band_center(rgb, frac=0.88, smooth=6):
    """per-column vertical centre (in rows of the given image) of the dark band (asphalt): weights = how much darker than frac*median luminance, rows restricted to the middle 70%"""
    lum = rgb[..., :3].astype(np.float32).mean(-1)
    H, W = lum.shape
    T = frac * np.median(lum)
    w = np.clip(T - lum, 0, None)
    w[: int(H * .15)] = 0; w[int(H * .85):] = 0
    ys = np.arange(H, dtype=np.float32)[:, None]
    c = (w * ys).sum(0) / np.maximum(w.sum(0), 1e-3)
    ok = w.sum(0) > 1e-2
    c = np.where(ok, c, np.nan)
    idx = np.arange(W)
    c = np.interp(idx, idx[ok], c[ok])
    return cv2.GaussianBlur(c[None, :].astype(np.float32), (0, 0), smooth)[0]


def follow_arc(gen_png, stem, out_png, smooth_o=0.8, smooth_g=40):
    """shift the generation's content vertically, column by column, so that its dark road band follows the ORIGINAL road's (curved) centre line"""
    g = np.asarray(Image.open(gen_png).convert("RGBA"))
    H, W = g.shape[:2]
    o = np.asarray(Image.open(udir(stem) / "orig.png").convert("RGBA"))
    co = band_center(o, 0.9, smooth_o) / o.shape[0]          # fraction of height, per original column
    co = np.interp(np.linspace(0, o.shape[1] - 1, W), np.arange(o.shape[1]), co)
    xs = np.linspace(0, 1, W)
    co = np.polyval(np.polyfit(xs, co, 3), xs)               # smooth arc: low-order fit
    cg = band_center(g, 0.8, 8) / H
    cg = np.polyval(np.polyfit(xs, cg, 3), xs)
    dy = (co - cg) * H                                        # px to shift down
    my = (np.arange(H, dtype=np.float32)[:, None] - dy[None, :]).astype(np.float32)
    mx = np.tile(np.arange(W, dtype=np.float32)[None, :], (H, 1))
    out = cv2.remap(g, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    Image.fromarray(out, "RGBA").save(out_png)
    return co, cg, dy


def rekey_zone(gen_dir, seed, zones, out_png):
    """finalize forces 'original-solid' areas opaque, which closes counters of small lettering (B, g, a, y ...) and smears it.  In the given zones (list of
    (x0,y0,x1,y1) fractions) use the plain soft chroma key of the colour-matched raw redraw instead."""
    import sys
    sys.path.insert(0, str(ROOT / "tools" / "pipeline"))
    from greenkey import green_key
    rgba = np.asarray(Image.open(f"{gen_dir}/klein_s{seed}_rgba.png").convert("RGBA")).copy()
    raw = np.asarray(Image.open(f"{gen_dir}/klein_s{seed}_cm.png").convert("RGB")).astype(np.float32)
    key, _, _ = green_key(raw)
    H, W = rgba.shape[:2]
    m = np.zeros((H, W), np.float32)
    for x0, y0, x1, y1 in zones:
        m[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = 1
    m = cv2.GaussianBlur(m, (0, 0), 3)[..., None]
    out = rgba.astype(np.float32) * (1 - m) + key.astype(np.float32) * m
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA").save(out_png)


def neon_alpha(cm_png, color_png, out_png, lo=45, hi=130, glow_sigma=9, glow_gain=0.9, size_ref=None):
    """crisp neon: alpha = max(tube mask from the redraw's brightness on its dark backing, soft glow = blurred tube mask).  Colours from color_png (neon_colors output).
    Replaces imposing the 8x-upscaled original alpha (which shows its source pixels)."""
    g = np.asarray(Image.open(cm_png).convert("RGB"), np.float32)
    v = g.max(-1)
    t = np.clip((v - lo) / (hi - lo), 0, 1)
    t = t * t * (3 - 2 * t)
    glow = np.clip(cv2.GaussianBlur(t, (0, 0), glow_sigma) * glow_gain, 0, 1)
    a = np.maximum(t, glow)
    rgb = np.asarray(Image.open(color_png).convert("RGB"), np.float32)
    Image.fromarray(np.dstack([rgb, a * 255]).astype(np.uint8), "RGBA").save(out_png)
