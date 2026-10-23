"""mag_icons.py: the three little icons that sit down the edge of two Hollywood Confidential covers (MAG3, MAG4) - a fan of four playing cards, a gold star, a skull and crossbones - drawn as vectors
(PIL at 16x, area-downsampled), so they are crisp at 4x instead of the original's 20 px blobs.  Shared parts: pipeline/mag_parts/_shared/{cards,star,skull}.png (RGBA, drawn at 8 px per
original px; place with fit_icon(), which finds scale / angle / position by matching the original's pixels).
  python tools/mag_icons.py   -> writes the shared canonical icons and a preview sheet"""
import math, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mag_parts as mp

SS = 16                                     # supersampling


def _canvas(w, h):
    return Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))


def _down(im, k):
    """canvas drawn at SS px/unit -> k px/unit"""
    return im.resize((max(1, round(im.width * k / SS)), max(1, round(im.height * k / SS))), Image.LANCZOS)


def _glow(im, dark=(18, 12, 8), alpha=0.45, grow=1.6):
    """the thin dark rim the originals have where an icon meets the page: a blurred dilated copy of the alpha under the icon"""
    pad = int(grow * 3 * SS)
    big = Image.new("RGBA", (im.width + 2 * pad, im.height + 2 * pad), (0, 0, 0, 0)); big.alpha_composite(im, (pad, pad)); im = big       # room for the rim round the edge
    a = np.asarray(im.getchannel("A")).astype(np.float32) / 255
    k = int(grow * SS) * 2 + 1
    d = cv2.GaussianBlur(cv2.dilate(a, np.ones((k, k), np.uint8)), (0, 0), grow * SS * 0.6) * alpha
    sh = Image.merge("RGBA", (Image.new("L", im.size, dark[0]), Image.new("L", im.size, dark[1]), Image.new("L", im.size, dark[2]), Image.fromarray((d * 255).astype(np.uint8))))
    sh.alpha_composite(im)
    return sh


def star(units=16.0):
    n = int(units)
    c = _canvas(n, n); d = ImageDraw.Draw(c)
    R, r, cx, cy = n * SS * 0.47, n * SS * 0.47 * 0.40, n * SS / 2, n * SS * 0.53
    pts = [(cx + (R if i % 2 == 0 else r) * math.cos(-math.pi / 2 + i * math.pi / 5), cy + (R if i % 2 == 0 else r) * math.sin(-math.pi / 2 + i * math.pi / 5)) for i in range(10)]
    d.polygon(pts, fill=(196, 160, 56, 255))                                    # darker rim
    k = 0.80
    pts2 = [(cx + (x - cx) * k, cy + (y - cy) * k) for x, y in pts]
    d.polygon(pts2, fill=(240, 218, 104, 255))
    k = 0.55
    d.polygon([(cx + (x - cx) * k, cy - (cy - y) * k - n * SS * 0.02) for x, y in pts], fill=(248, 232, 130, 255))   # lighter centre
    return c


def skull(units=24.0):
    w, h = 24, 24
    c = _canvas(w, h); d = ImageDraw.Draw(c)
    s = SS * w / 100.0
    def P(x, y): return (x * s, y * s)
    def bone(x0, y0, x1, y1):
        ang = math.atan2(y1 - y0, x1 - x0); nx, ny = -math.sin(ang), math.cos(ang)
        d.line([P(x0, y0), P(x1, y1)], fill=(236, 230, 214, 255), width=int(7.5 * s))
        for (ex, ey) in ((x0, y0), (x1, y1)):
            for sgn in (-1, 1):
                cx, cy = ex + nx * 4.2 * sgn, ey + ny * 4.2 * sgn
                d.ellipse([P(cx - 5.2, cy - 5.2), P(cx + 5.2, cy + 5.2)], fill=(240, 235, 220, 255))
        d.line([P(x0, y0), P(x1, y1)], fill=(206, 198, 180, 255), width=int(1.6 * s))
    bone(14, 66, 86, 93); bone(86, 66, 14, 93)
    d.ellipse([P(18, 6), P(82, 62)], fill=(250, 248, 240, 255))                  # cranium
    d.rounded_rectangle([P(30, 46), P(70, 74)], radius=int(8 * s), fill=(246, 243, 233, 255))   # jaw
    d.ellipse([P(26, 30), P(46, 54)], fill=(14, 10, 10, 255)); d.ellipse([P(54, 30), P(74, 54)], fill=(14, 10, 10, 255))     # eye sockets
    d.polygon([P(50, 52), P(44, 64), P(56, 64)], fill=(30, 24, 22, 255))        # nose
    for x in (42, 50, 58):
        d.line([P(x, 67), P(x, 74)], fill=(150, 142, 128, 255), width=int(1.4 * s))
    d.arc([P(18, 6), P(82, 62)], 200, 340, fill=(215, 208, 192, 255), width=int(1.5 * s))
    return c


def _card(kind, w=60, h=84):
    c = _canvas(w, h); d = ImageDraw.Draw(c)
    s = SS
    def P(x, y): return (x * s, y * s)
    d.rounded_rectangle([P(0, 0), P(w, h)], radius=int(6 * s), fill=(238, 236, 232, 255))
    b = 4
    inner = [P(b, b), P(w - b, h - b)]
    if kind == "plane":                                                           # yellow card: red aeroplane over a teal beam
        d.rounded_rectangle(inner, radius=int(3 * s), fill=(236, 222, 52, 255))
        d.polygon([P(24, 44), P(36, 44), P(46, h - b - 1), P(14, h - b - 1)], fill=(36, 160, 200, 255))
        d.polygon([P(30, 16), P(34, 16), P(34, 40), P(30, 40)], fill=(200, 28, 40, 255))
        d.polygon([P(14, 28), P(50, 26), P(50, 33), P(14, 35)], fill=(200, 28, 40, 255))
        d.polygon([P(24, 40), P(40, 40), P(36, 46), P(28, 46)], fill=(200, 28, 40, 255))
    elif kind == "sun":                                                           # lavender-grey card with a gold disc
        d.rounded_rectangle(inner, radius=int(3 * s), fill=(176, 172, 190, 255))
        d.ellipse([P(10, 18), P(50, 66)], fill=(214, 198, 24, 255)); d.ellipse([P(14, 22), P(46, 62)], fill=(232, 218, 40, 255))
    elif kind == "queen":                                                         # red card with a yellow-haired face
        d.rounded_rectangle(inner, radius=int(3 * s), fill=(206, 24, 44, 255))
        d.ellipse([P(12, 14), P(48, 66)], fill=(244, 214, 40, 255))
        d.ellipse([P(20, 24), P(40, 54)], fill=(240, 214, 190, 255))
        d.polygon([P(8, h - 8), P(52, h - 8), P(52, h - b), P(8, h - b)], fill=(190, 20, 38, 255))
    else:                                                                         # red and white card with a black-and-white disc
        d.rounded_rectangle(inner, radius=int(3 * s), fill=(206, 24, 44, 255))
        d.ellipse([P(14, 24), P(46, 52)], fill=(236, 234, 230, 255)); d.ellipse([P(30, 26), P(48, 46)], fill=(18, 14, 14, 255))
        d.polygon([P(8, 58), P(52, 58), P(52, h - b), P(8, h - b)], fill=(230, 232, 230, 255))
        d.polygon([P(12, 62), P(48, 62), P(48, h - b - 2), P(12, h - b - 2)], fill=(206, 24, 44, 255))
    return c


def cards(units_w=60, units_h=44, angles=(-44, -16, 12, 40), spread=1.0, arm=26):
    """the fan: four cards turned about a common pivot below the middle of the fan (pivot at the bottom of the canvas)"""
    kinds = ("plane", "sun", "queen", "flag")
    W, H = 170, 150
    out = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    px, py = W / 2 * SS, 140 * SS
    for kind, a in sorted(zip(kinds, angles), key=lambda ka: ("sun", "flag", "queen", "plane").index(ka[0])):     # back to front
        card = _card(kind)
        cw, ch = card.size
        # card bottom-centre ARM units above the pivot, then turned by a about the pivot
        layer = Image.new("RGBA", out.size, (0, 0, 0, 0)); layer.paste(card, (int(px - cw / 2), int(py - ch - arm * SS)))
        layer = layer.rotate(-a * spread, resample=Image.BICUBIC, center=(px, py))
        sh = _glow(layer, alpha=0.5, grow=1.0)
        out.alpha_composite(sh)
    return out


def tight(im):
    bb = im.getbbox()
    return im.crop(bb) if bb else im


def render(name, k):
    """icon NAME as RGBA at k px per unit"""
    c = {"star": star, "skull": skull, "cards": cards}[name]()
    return tight(_down(_glow(c) if name != "cards" else c, k))


def tint_to(icon, target_rgb, k=1.0):
    """per-channel gains so the icon's mean colour (under its alpha) equals TARGET_RGB (the original's mean colour at the fitted spot), clipped"""
    a = np.asarray(icon).astype(np.float32)
    w = a[..., 3] / 255
    mean = (a[..., :3] * w[..., None]).sum((0, 1)) / max(w.sum(), 1e-6)
    g = np.clip(np.asarray(target_rgb, np.float32) / np.maximum(mean, 1), 0.45, 1.35) ** k
    a[..., :3] = np.clip(a[..., :3] * g, 0, 255)
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def fit_icon(orig_rgb, name, approx, scales=np.linspace(0.7, 1.5, 17), angles=range(-12, 13, 3), base_k=None):
    """find where / how big / how turned ICON NAME sits in the original: NCC of the icon's grey (alpha-masked) against the original around APPROX = (x0, y0, x1, y1) in orig px.
    Returns dict(scale, angle, x, y (top-left of the rotated icon, orig px, float), score)"""
    x0, y0, x1, y1 = approx
    base_k = base_k or {"cards": 0.205, "star": 1.0, "skull": 1.0}[name]
    g = cv2.cvtColor(np.asarray(orig_rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
    region = g[y0:y1, x0:x1]
    best = None
    base = {"star": star, "skull": skull, "cards": cards}[name]()
    base = _glow(base) if name != "cards" else base
    for s in scales:
        ic = tight(_down(base, base_k * s * 4))                                    # 4 px per orig px, then matched at orig scale below
        for a in angles:
            r = ic.rotate(a, resample=Image.BICUBIC, expand=True)
            r = tight(r)
            t = r.resize((max(2, round(r.width / 4)), max(2, round(r.height / 4))), Image.LANCZOS)
            if t.width >= region.shape[1] or t.height >= region.shape[0]:
                continue
            ta = np.asarray(t).astype(np.float32)
            tg = cv2.cvtColor(ta[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            m = (ta[..., 3] / 255).astype(np.float32)
            res = cv2.matchTemplate(region, tg, cv2.TM_CCORR_NORMED, mask=np.dstack([m] * 1)[..., 0]) if False else cv2.matchTemplate(region, tg * m, cv2.TM_CCORR_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if best is None or mx > best["score"]:
                best = {"scale": float(s), "angle": a, "x": x0 + loc[0], "y": y0 + loc[1], "score": float(mx), "w": t.width, "h": t.height}
    return best


def refine_icon(orig_rgb, name, fit, k=4, pad=7):
    """sub-pixel pass of fit_icon: region and icon at K x, small search in scale / angle / position -> dict(x, y at K x page scale, scale, angle, score) + the rendered RGBA (tight)"""
    base_k = {"cards": 0.205, "star": 1.0, "skull": 1.0}[name]
    base = {"star": star, "skull": skull, "cards": cards}[name]()
    base = _glow(base) if name != "cards" else base
    x0, y0 = int(fit["x"]) - pad, int(fit["y"]) - pad
    x1, y1 = int(fit["x"] + fit["w"]) + pad, int(fit["y"] + fit["h"]) + pad
    reg = Image.fromarray(np.asarray(orig_rgb)[y0:y1, x0:x1]).resize(((x1 - x0) * k, (y1 - y0) * k), Image.LANCZOS)
    g = cv2.cvtColor(np.asarray(reg), cv2.COLOR_RGB2GRAY).astype(np.float32)
    best = None
    for ds in (0.94, 0.97, 1.0, 1.03, 1.06):
        s = fit["scale"] * ds
        ic = tight(_down(base, base_k * s * k))
        for a in (fit["angle"] - 2, fit["angle"], fit["angle"] + 2):
            r = tight(ic.rotate(a, resample=Image.BICUBIC, expand=True))
            if r.width >= g.shape[1] or r.height >= g.shape[0]:
                continue
            ra = np.asarray(r).astype(np.float32)
            tg = cv2.cvtColor(ra[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) * (ra[..., 3] / 255)
            _, mx, _, loc = cv2.minMaxLoc(cv2.matchTemplate(g, tg, cv2.TM_CCORR_NORMED))
            if best is None or mx > best[0]:
                best = (mx, s, a, loc, r)
    mx, s, a, loc, r = best
    return {"score": float(mx), "scale": float(s), "angle": a, "x": x0 * k + loc[0], "y": y0 * k + loc[1]}, r


def icon_part(orig_rgb, name, fit, tint=True):
    """the icon placed where the original has it: (RGBA at 4x page scale, x, y, info).  Tinted so its mean colour equals the original's under the same footprint"""
    info, r = refine_icon(orig_rgb, name, fit)
    x, y = int(round(info["x"])), int(round(info["y"]))
    if tint:
        big = np.asarray(Image.fromarray(np.asarray(orig_rgb)).resize((orig_rgb.shape[1] * 4, orig_rgb.shape[0] * 4), Image.LANCZOS))
        patch = big[y:y + r.height, x:x + r.width].astype(np.float32)
        w = np.asarray(r.getchannel("A")).astype(np.float32) / 255
        if patch.shape[:2] == w.shape:
            tgt = (patch * w[..., None]).sum((0, 1)) / max(w.sum(), 1e-6)
            r = tint_to(r, tgt, 0.8)
    return r, x, y, info


if __name__ == "__main__":
    out = mp.PARTS / "_shared"; out.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGBA", (900, 330), (90, 90, 90, 255)); x = 10
    for n in ("cards", "star", "skull"):
        im = render(n, 8); im.save(out / f"{n}.png"); sheet.alpha_composite(im, (x, 10)); x += im.width + 30
    sheet.convert("RGB").save(mp.PARTS / "_shared" / "icons_preview.png"); print("icons written")
