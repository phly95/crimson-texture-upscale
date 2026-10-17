"""texlab: helpers the texture agents use. Import with:

    import sys; sys.path.insert(0, "/home/deck/workspace/csvm-hd/tools")
    from texlab import *

Everything works on PIL images (RGB or RGBA, 8-bit). Nothing here touches the game install.
"""
import io, json, os, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from skimage import color as skcolor, exposure
import cv2
import warnings
warnings.filterwarnings("ignore", message=".*negative Z values.*")   # harmless LAB->sRGB gamut clipping

ROOT = Path(__file__).resolve().parents[1]
Image.MAX_IMAGE_PIXELS = 100_000_000
_cat = None
_lib = None


# ---------------------------------------------------------------- catalog / originals
def catalog():
    global _cat
    if _cat is None:
        _cat = json.load(open(ROOT / "catalog.json"))
    return _cat


def info(uid):
    return catalog()["units"][uid]


def orig_path(uid):
    u = info(uid)
    ch = u["stems"][u["stem"]][0]
    return ROOT / "originals" / ch / f"{u['stem']}.png"


def orig(uid, mode="RGBA"):
    """The original texture as it ships (small!). Use up() to enlarge it."""
    return Image.open(orig_path(uid)).convert(mode)


# ---------------------------------------------------------------- open-asset library
def lib_index():
    global _lib
    if _lib is None:
        _lib = [json.loads(l) for l in open(ROOT / "assets" / "index.jsonl")]
    return _lib


def lib_rec(asset_id):
    asset_id = asset_id.split(":")[-1]
    for r in lib_index():
        if r["id"] == asset_id:
            return r
    raise KeyError(f"unknown library asset {asset_id!r}")


def lib_search(text="", near=None, n=24, source=None, min_side=0):
    """Rank library assets. `text`: words matched against name/category/tags (all must hit at
    least one field; more hits rank higher). `near`: (r,g,b) 0-255, ranks by mean-colour distance
    among the matches. Returns records (dicts with id, name, category, mean_rgb, w, h, license)."""
    words = [w for w in text.lower().replace(",", " ").split() if w]
    out = []
    for r in lib_index():
        if source and r["source"] != source:
            continue
        if min(r["w"], r["h"]) < min_side:
            continue
        hay = " ".join([r["name"], r["category"]] + r["tags"]).lower()
        hits = sum(1 for w in words if w in hay)
        if words and hits == 0:
            continue
        d = 0.0
        if near is not None:
            m = r["mean_rgb"]
            d = sum((a - b) ** 2 for a, b in zip(m, near)) ** 0.5
        out.append((-hits, d, r))
    out.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in out[:n]]


def lib_image(asset_id, size=None, mode="RGB"):
    r = lib_rec(asset_id)
    im = Image.open(ROOT / r["path"]).convert(mode)
    if size:
        im = im.resize((size, size) if isinstance(size, int) else tuple(size), Image.LANCZOS)
    return im


def lib_sheet(recs, path, cell=176, cols=6):
    """Contact sheet of library assets with their ids, for viewing with the Read tool."""
    rows = (len(recs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell + 6), rows * (cell + 20)), (30, 30, 36))
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(recs):
        im = Image.open(ROOT / r["path"]).convert("RGB")
        im.thumbnail((cell, cell))
        x, y = (i % cols) * (cell + 6), (i // cols) * (cell + 20)
        sheet.paste(im, (x, y))
        d.text((x, y + cell + 3), r["id"][:26], fill=(235, 235, 235))
    sheet.save(path)
    return path


# ---------------------------------------------------------------- basic image ops
def up(img, size, kind="lanczos"):
    """Smooth resize to size=(w,h) or int (square). kind: lanczos|cubic|linear|nearest."""
    if isinstance(size, int):
        size = (size, size)
    k = {"lanczos": Image.LANCZOS, "cubic": Image.BICUBIC, "linear": Image.BILINEAR,
         "nearest": Image.NEAREST, "box": Image.BOX}[kind]
    return img.resize(tuple(size), k)


def to_f(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def from_f(a):
    return Image.fromarray(np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")


def blur(img, sigma):
    return from_f(cv2.GaussianBlur(to_f(img), (0, 0), sigma))


def match_color(img, ref, strength=1.0, mode="meanstd", ref_mask=None):
    """Recolour `img` to look like `ref` (any sizes). mode: 'meanstd' (LAB mean/std transfer, good
    default), 'hist' (full histogram match), 'mean' (shift only). strength 0..1 blends toward the
    result. ref_mask: optional boolean array/PIL 'L' same size as ref marking pixels to use."""
    a = to_f(img)
    b = to_f(ref)
    if ref_mask is not None:
        m = np.asarray(ref_mask.convert("L") if hasattr(ref_mask, "convert") else ref_mask) > 127
        b = b[m.reshape(b.shape[:2])] if m.any() else b.reshape(-1, 3)
        b = b.reshape(-1, 1, 3)
    if mode == "hist":
        out = exposure.match_histograms(a, b.reshape(-1, 1, 3), channel_axis=-1)
    else:
        la, lb = skcolor.rgb2lab(a), skcolor.rgb2lab(b)
        ma, sa = la.reshape(-1, 3).mean(0), la.reshape(-1, 3).std(0) + 1e-6
        mb, sb = lb.reshape(-1, 3).mean(0), lb.reshape(-1, 3).std(0) + 1e-6
        lab = la - ma
        if mode == "meanstd":
            lab = lab * (sb / sa)
        out = skcolor.lab2rgb(lab + mb)
    return from_f(a * (1 - strength) + out * strength)


def inject_detail(base, detail, sigma, gain=1.0):
    """Frequency separation: keep `base`'s colours/layout below `sigma` px, take everything finer
    from `detail` (same size). The core trick for keeping the original's look but adding crisp
    texture: base=up(original, N), detail=colour-matched library crop. sigma ~ N/32 .. N/12."""
    if base.size != detail.size:
        detail = detail.resize(base.size, Image.LANCZOS)
    b, d = to_f(base), to_f(detail)
    low_b = cv2.GaussianBlur(b, (0, 0), sigma)
    low_d = cv2.GaussianBlur(d, (0, 0), sigma)
    return from_f(low_b + gain * (d - low_d))


def bleed_rgb(img, thresh=128):
    """Return an RGB image of `img` with the colour under transparent pixels (alpha < thresh)
    inpainted from neighbours. Original sprites store black under alpha=0, which smears dark
    fringes into the edges when enlarged: call this before up()/up_wrap() on anything with alpha."""
    a = np.asarray(img.convert("RGBA"))
    rgb = np.ascontiguousarray(a[..., :3])
    m = a[..., 3] < thresh
    if m.any():
        rgb = cv2.inpaint(rgb, m.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(rgb, "RGB")


def _pad2(f, p, wrap):
    f = np.pad(f, ((p, p), (0, 0), (0, 0)), mode="wrap" if wrap[1] else "edge")
    return np.pad(f, ((0, 0), (p, p), (0, 0)), mode="wrap" if wrap[0] else "edge")


def up_wrap(img, size, wrap=(False, False), crisp=(5.0, 1.3, 6)):
    """Bicubic enlargement that is aware of tiling (wrap=(x,y): pad by wrapping instead of clamping,
    so a seamless tile stays seamless; a plain resize multiplies the seam ratio ~8x) followed by a
    luminance-only unsharp clamped to the local min/max (crisp edges, no colour fringes, no halos).
    size must be an integer multiple of the source size. crisp=(sigma, amount, clamp_radius) or None."""
    if isinstance(size, int):
        size = (size, size)
    f = to_f(img)
    h, w, _ = f.shape
    tw, th = size
    assert tw % w == 0 and th % h == 0, "up_wrap needs an integer scale"
    sx, sy, p = tw // w, th // h, 4
    big = np.clip(cv2.resize(_pad2(f, p, wrap), ((w + 2 * p) * sx, (h + 2 * p) * sy), interpolation=cv2.INTER_CUBIC), 0, 1)
    if crisp and crisp[1] > 0:
        lab = skcolor.rgb2lab(big).astype(np.float32)
        L = lab[..., 0]
        sh = L + crisp[1] * (L - cv2.GaussianBlur(L, (0, 0), crisp[0]))
        k = np.ones((2 * crisp[2] + 1, 2 * crisp[2] + 1), np.uint8)
        lab[..., 0] = np.clip(sh, cv2.erode(L, k), cv2.dilate(L, k))
        big = np.clip(skcolor.lab2rgb(lab), 0, 1).astype(np.float32)
    return from_f(big[p * sy:p * sy + th, p * sx:p * sx + tw])


def inject_detail_lab(base, detail, sigma, rel=0.18, cfac=0.35, fade_L=28.0, mask=None, wrap=(False, False)):
    """Detail injection in LAB, gentler than inject_detail (which adds RGB noise): the asset's fine
    luminance detail MODULATES the base multiplicatively (black stays black, hue and layout stay the
    original's) and a little of its fine chroma is added, faded out in the shadows.
    base/detail: same-size images (detail is resized if not). sigma: px below which detail comes from
    the asset (~target/64 .. target/24). rel: luminance strength (0.1-0.3 typical). cfac: chroma
    strength. mask: optional float array 0..1 (h,w) for where detail applies (region-specific assets)."""
    if detail.size != base.size:
        detail = detail.resize(base.size, Image.LANCZOS)
    lb = skcolor.rgb2lab(to_f(base)).astype(np.float32)
    ld = skcolor.rgb2lab(to_f(detail)).astype(np.float32)
    p = int(sigma * 3) + 2
    low = cv2.GaussianBlur(_pad2(ld, p, wrap), (0, 0), sigma)[p:-p, p:-p]
    hf = ld - low
    hfl = np.clip(hf[..., 0] / (hf[..., 0].std() + 1e-6), -3.5, 3.5)
    m = np.ones(lb.shape[:2], np.float32) if mask is None else mask.astype(np.float32)
    out = lb.copy()
    out[..., 0] = np.clip(lb[..., 0] * (1 + rel * hfl * m), 0, 100)
    fade = np.clip(lb[..., 0] / fade_L, 0, 1)[..., None]
    out[..., 1:] = lb[..., 1:] + cfac * hf[..., 1:] * m[..., None] * fade
    return from_f(skcolor.lab2rgb(out))


def make_tileable(img, border=0.5):
    """Two-pass cross-fade against a half-rolled copy; result wraps seamlessly. border 0..1 is how
    much of the image the fade covers (0.5 = classic)."""
    a = to_f(img)
    h, w, _ = a.shape

    def ramp(n):
        x = (np.arange(n) + 0.5) / n
        t = np.clip(np.abs(x - 0.5) * 2 / max(border, 1e-3), 0, 1)       # 0 at centre, 1 at/after edge
        return 0.5 + 0.5 * np.cos(np.pi * t)                             # 1 at centre, 0 at edges
    ax = ramp(w)[None, :, None]
    out = a * ax + np.roll(a, w // 2, axis=1) * (1 - ax)
    ay = ramp(h)[:, None, None]
    out = out * ay + np.roll(out, h // 2, axis=0) * (1 - ay)
    return from_f(out)


def region(asset, size, frac=0.5, seed=0, flip=False):
    """Random crop of `asset` with the aspect ratio of `size` (int or (w,h)), covering `frac` of
    the asset's width (smaller frac = more zoomed in = coarser-looking detail), resized to size."""
    if isinstance(size, int):
        size = (size, size)
    rng = np.random.default_rng(seed)
    w, h = asset.size
    aspect = size[0] / size[1]
    cw = max(8, int(w * frac))
    ch = int(round(cw / aspect))
    if ch > h:
        ch = h
        cw = int(round(ch * aspect))
    x = int(rng.integers(0, max(1, w - cw + 1)))
    y = int(rng.integers(0, max(1, h - ch + 1)))
    c = asset.crop((x, y, x + cw, y + ch)).resize(tuple(size), Image.LANCZOS)
    return c.transpose(Image.FLIP_LEFT_RIGHT) if flip else c


def region_stretched(asset, size, frac=0.5, seed=0, stretch=4.0, axis="y", flip=False):
    """Anisotropic version of region() for streaky material (roads, worn dirt, brushed metal, wood):
    crops the asset with one axis shortened, then stretches it by `stretch` on the way to `size`, so
    an isotropic grit photo becomes lengthwise streaks. axis is the direction the STREAKS RUN in the
    output ("y" = vertical streaks, grain 90deg). Measured on gravelroad: rocks_ground_03, frac 0.9,
    stretch 4 gives grain 90deg/0.74 vs the original's 90deg/0.75."""
    if isinstance(size, int):
        size = (size, size)
    rng = np.random.default_rng(seed)
    w, h = asset.size
    aspect = size[0] / size[1]
    if axis == "y":
        aspect = aspect / stretch            # short crop -> stretched tall
    else:
        aspect = aspect * stretch
    cw = max(8, int(w * frac))
    ch = int(round(cw / aspect))
    if ch > h:
        ch = h
        cw = int(round(ch * aspect))
    if cw > w:
        cw = w
        ch = int(round(cw / aspect))
    x = int(rng.integers(0, max(1, w - cw + 1)))
    y = int(rng.integers(0, max(1, h - ch + 1)))
    c = asset.crop((x, y, x + cw, y + ch)).resize(tuple(size), Image.LANCZOS)
    return c.transpose(Image.FLIP_LEFT_RIGHT) if flip else c


def pchip_up(img, size):
    """Monotone cubic (PCHIP) enlargement: follows the original's gradients without the ringing that
    Lanczos/bicubic put around hard edges. The right base for smooth art: glows, flares, gauges,
    gradients, chrome. Accepts and returns a PIL image."""
    from scipy.interpolate import PchipInterpolator
    if isinstance(size, int):
        size = (size, size)
    a = to_f(img).astype(np.float32)
    h, w, _ = a.shape
    xs = (np.arange(size[0]) + 0.5) * w / size[0] - 0.5
    ys = (np.arange(size[1]) + 0.5) * h / size[1] - 0.5
    a = PchipInterpolator(np.arange(w), a, axis=1)(np.clip(xs, 0, w - 1))
    a = PchipInterpolator(np.arange(h), a, axis=0)(np.clip(ys, 0, h - 1))
    return from_f(np.clip(a, 0, 1))


def ibp(a, size, iters=20):
    """Iterative back-projection reconstruction of a 1-channel float array (0..1): start from a
    cubic enlargement, then repeatedly add the residual (original - area-shrunk current). Converges
    to an enlargement that shrinks back to the original almost exactly - the analytic alternative to
    a neural upscaler for smooth/soft content (sprite alpha, glows, gradients). Returns float32."""
    h, w = a.shape
    if isinstance(size, int):
        size = (size, size)
    H = cv2.resize(a.astype(np.float32), tuple(size), interpolation=cv2.INTER_CUBIC)
    for _ in range(iters):
        r = a.astype(np.float32) - cv2.resize(H, (w, h), interpolation=cv2.INTER_AREA)
        H = np.clip(H + cv2.resize(r, tuple(size), interpolation=cv2.INTER_CUBIC), 0, 1)
    return H


def alpha_convention(img):
    """'premult' if the RGB is dark where the alpha is zero (the sprite is stored premultiplied-style,
    like the splash/fire frames: rebuilding RGB as colour_ratio*alpha is then mandatory) or 'straight'
    if the colour continues under the transparent area (flares, glows, indicators: enlarge the RGB on
    its own and leave it alone)."""
    a = np.asarray(img.convert("RGBA"), np.float32) / 255.0
    lo, hi = a[a[..., 3] < 0.05], a[a[..., 3] > 0.90]
    if len(lo) < 4 or len(hi) < 4:
        return "straight"
    return "premult" if lo[..., :3].mean() < 0.35 * max(hi[..., :3].mean(), 1e-3) else "straight"


def soft_up(img, size, iters=20):
    """Enlargement built for soft sprites (flares, glows, smoke, shadows, indicator lamps): PCHIP
    colour (no ringing on gradients) plus back-projected alpha so the shrunk silhouette matches the
    original exactly. Handles both storage conventions - see alpha_convention(): a premultiplied
    sprite is rebuilt as colour_ratio*new_alpha, a straight one keeps its own RGB. For content that
    needs real new detail (a logo, a hatch pattern) redraw it instead; this only reconstructs."""
    if isinstance(size, int):
        size = (size, size)
    a = np.asarray(img.convert("RGBA"), np.float32) / 255.0
    A0 = a[..., 3]
    soft = np.count_nonzero((A0 > 0.02) & (A0 < 0.98))
    A = ibp(A0, size, iters) if soft or A0.max() < 1.0 else \
        cv2.resize(A0, tuple(size), interpolation=cv2.INTER_CUBIC)
    if alpha_convention(img) == "premult":
        vis = A0 > 0.15
        col = np.where(vis[..., None], a[..., :3], np.asarray(a[..., :3][vis]).mean((0, 1))) if vis.sum() > 4 else a[..., :3]
        ratio = np.clip(col / np.maximum(A0[..., None], 0.08), 0, 1)      # un-premultiply
        ratio = np.asarray(pchip_up(from_f(ratio), size), np.float32) / 255.0
        if vis.sum() > 4:                                                 # colour under alpha~0 is guesswork
            mr = np.clip(a[..., :3][vis].mean(0) / max(A0[vis].mean(), 1e-3), 0, 1)
            wv = np.clip((A - 0.04) / 0.20, 0, 1)[..., None]
            ratio = ratio * wv + np.asarray(mr)[None, None, :] * (1 - wv)
        rgb = np.clip(ratio, 0, 1) * A[..., None]
    else:
        rgb = np.asarray(pchip_up(from_f(a[..., :3]), size), np.float32) / 255.0
    out = np.dstack([rgb, A])
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")


def grain(asset_id, size, amp=0.03, seed=0, zoom=1.0, vstreak=0.0, hstreak=0.0):
    """Fine surface texture as a multiplicative field (mean 0, unit std, already scaled by `amp`),
    from a library asset's own high frequencies: add as img * (1 + grain(...)). vstreak/hstreak
    smear it along one axis (brushed metal, wood, one-directional grit). Use amp 0.02-0.05: enough to
    read as material at 1:1, not enough to fight the original's own pattern."""
    if isinstance(size, int):
        size = (size, size)
    a = lib_image(asset_id)
    c = region(a, size, frac=min(0.9, size[0] * zoom / max(a.size)), seed=seed)
    g = np.asarray(c.convert("L"), np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 2.5)
    if vstreak:
        g = 0.5 * g + 0.5 * cv2.GaussianBlur(g, (0, 0), sigmaX=0.6, sigmaY=vstreak)
    if hstreak:
        g = 0.5 * g + 0.5 * cv2.GaussianBlur(g, (0, 0), sigmaX=hstreak, sigmaY=0.6)
    return (g / (g.std() + 1e-6) * amp).astype(np.float32)


def alpha_from_orig(uid, size, cutout=False):
    """Alpha channel of the original, enlarged. cutout=True keeps it hard-edged (fences, trees,
    cut-out decals): bicubic then threshold with a 1px soft edge. Otherwise smooth bicubic."""
    a = orig(uid).getchannel("A")
    a = a.resize(tuple(size) if not isinstance(size, int) else (size, size), Image.BICUBIC)
    if cutout:
        arr = np.asarray(a, dtype=np.float32)
        arr = np.clip((arr - 128.0) * 12.0 + 128.0, 0, 255)
        a = Image.fromarray(arr.astype(np.uint8), "L")
    return a


# ---------------------------------------------------------------- vector helpers
def render_svg(svg_text, size):
    """Rasterize SVG text to an RGBA image of size (w,h) or int."""
    import cairosvg
    if isinstance(size, int):
        size = (size, size)
    png = cairosvg.svg2png(bytestring=svg_text.encode(), output_width=size[0], output_height=size[1])
    return Image.open(io.BytesIO(png)).convert("RGBA")


def grid_view(uid, path, scale=None, step=8):
    """Write the original enlarged (nearest) with a labelled pixel-coordinate grid, for drawing SVG
    by hand: read coordinates off it and write the SVG with viewBox="0 0 W H" in the ORIGINAL's pixel
    units (render_svg then scales it to any size). Transparent areas show magenta. Read the file."""
    o = orig(uid)
    if scale is None:
        scale = max(2, min(1024 // o.width, 1024 // o.height, 16))
    big = o.resize((o.width * scale, o.height * scale), Image.NEAREST)
    bg = Image.new("RGBA", big.size, (255, 0, 255, 255))
    bg.alpha_composite(big)
    d = ImageDraw.Draw(bg)
    for x in range(0, o.width + 1, step):
        d.line([(x * scale, 0), (x * scale, bg.height)], fill=(255, 255, 255, 110), width=1)
        d.text((x * scale + 2, 2), str(x), fill=(255, 255, 0, 255))
    for y in range(0, o.height + 1, step):
        d.line([(0, y * scale), (bg.width, y * scale)], fill=(255, 255, 255, 110), width=1)
        d.text((2, y * scale + 2), str(y), fill=(255, 255, 0, 255))
    bg.convert("RGB").save(path)
    return path


def svg_try(uid, svg_text, path=None):
    """Render hand-written SVG at the unit's target size and, if `path` is given, save a side-by-side
    (original enlarged | your SVG | difference heat-map) to compare. Returns the RGBA render.
    Tip: draw in the original's pixel units (viewBox="0 0 W H"); keep the same silhouette and palette."""
    o = orig(uid)
    tw, th = info(uid)["target"]
    r = render_svg(svg_text, (tw, th))
    if path:
        P = 384
        k = min(P / tw, P / th)
        sz = (max(1, int(tw * k)), max(1, int(th * k)))
        def over(im):
            bg = Image.new("RGBA", sz, (255, 0, 255, 255))
            bg.alpha_composite(im.resize(sz, Image.LANCZOS if im.width > sz[0] else Image.NEAREST))
            return bg.convert("RGB")
        a, b = over(o.resize((tw, th), Image.BICUBIC)), over(r)
        diff = Image.fromarray(np.clip(np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).sum(2) * 2, 0, 255).astype(np.uint8))
        sheet = Image.new("RGB", (sz[0] * 3 + 8, sz[1]), (30, 30, 36))
        sheet.paste(a, (0, 0)); sheet.paste(b, (sz[0] + 4, 0)); sheet.paste(diff.convert("RGB"), (2 * sz[0] + 8, 0))
        sheet.save(path)
    return r


def vectorize(img, size, pre=4):
    """Trace a flat-colour original (logo, glyph, gauge, cartoon decal) into smooth curves and
    render at `size`. The source is enlarged `pre`x first so curves come out smooth. Runs vtracer
    in a subprocess with its default settings (its option arguments crash on this machine). Good
    for flat art, poor for photos/gradients. Returns RGBA; fix alpha with alpha_from_orig."""
    if isinstance(size, int):
        size = (size, size)
    big = img.convert("RGBA").resize((img.width * pre, img.height * pre), Image.LANCZOS)
    with tempfile.TemporaryDirectory() as td:
        src, dst = os.path.join(td, "in.png"), os.path.join(td, "out.svg")
        big.save(src)
        code = f"import vtracer;vtracer.convert_image_to_svg_py({src!r},{dst!r})"
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, timeout=120)
        if r.returncode != 0 or not os.path.exists(dst):
            raise RuntimeError("vtracer failed (exit %s)" % r.returncode)
        svg = open(dst).read()
    return render_svg(svg, size)


def quantize(img, colors=16):
    """Reduce to N colours (median cut) - handy to snap a result back to a flat painted look."""
    q = img.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.NONE)
    return q.convert("RGB")


# ---------------------------------------------------------------- grain / structure
# ---------------------------------------------------------------- "seeing" without an image decoder
# This environment cannot show images to the agent (the Read tool has no image encoder), so these
# render an image as TEXT the agent can actually read: a density map of luminance or edges, with the
# aspect ratio kept. Structure you can read in ASCII: streaks and their direction, pattern scale,
# whether a design's lines and blobs survived, where the seams are.
_RAMP = " .`'-,:;!*+@#"


def _as_img(x):
    if isinstance(x, str):
        x = orig(x) if "__" in x and not Path(x).exists() else Image.open(Path(x).expanduser())
    return x


def text_view(x, cols=64, mode="lum", alpha_bg=0.35, stretch=True):
    """ASCII portrait of a uid, path or image. mode: 'lum' luminance (shape and colour layout),
    'edge' Sobel magnitude (the design's lines and the scale of its pattern), 'sat' saturation map.
    Transparent pixels are composited over mid-grey. `stretch` renormalises 2%-98% of the values onto
    the whole ramp, which is what makes a low-contrast painted texture readable (set False to compare
    absolute brightness between two images). Returns a string - print it."""
    im = _as_img(x).convert("RGBA")
    a = np.asarray(im, np.float32) / 255.0
    al = a[..., 3:4]
    rgb = a[..., :3] * al + alpha_bg * (1 - al)
    if mode == "edge":
        g = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        v = np.maximum(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), 0) + np.maximum(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3), 0)
        v = v / (np.percentile(v, 97) + 1e-6)
    elif mode == "sat":
        hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        v = hsv[..., 1].astype(np.float32) / 255.0
    else:
        v = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
        if stretch:
            lo, hi = np.percentile(v, [2, 98])
            v = (v - lo) / max(hi - lo, 1e-6)
    h, w = v.shape
    rows = max(2, int(round(cols * h / w / 2.0)))              # chars are ~2x taller than wide
    small = cv2.resize(v, (cols, rows), interpolation=cv2.INTER_AREA)
    small = np.clip(small, 0, 1)
    idx = np.clip((small * (len(_RAMP) - 1)).astype(int), 0, len(_RAMP) - 1)
    lines = ["".join(_RAMP[c] for c in row) for row in idx]
    pal = palette(im, 5)
    hdr = (f"{mode} view {w}x{h} -> {cols}x{rows}  ramp dark->light '{_RAMP}'  colours: "
           + "  ".join(f"{c} {100 * n / (im.width * im.height):.0f}%" for c, n in pal))
    return hdr + "\n" + "\n".join(lines)


def palette(im, k=6):
    """Dominant colours (4-bit per channel) with their share of the pixels, as #rrggbb."""
    a = np.asarray(_as_img(im).convert("RGB"), np.uint8)
    q = (a.reshape(-1, 3) >> 4).astype(np.uint16)
    code = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    vals, cnts = np.unique(code, return_counts=True)
    out = []
    for v, c in sorted(zip(vals, cnts), key=lambda t: -t[1])[:k]:
        r, g, b = (v >> 8) * 16 + 8, ((v >> 4) & 15) * 16 + 8, (v & 15) * 16 + 8
        out.append(("#%02x%02x%02x" % (r, g, b), int(c)))
    return out


def stats_view(x, label=""):
    """One readable line of the numbers the gate cares about, for any image or uid."""
    im = _as_img(x)
    a = np.asarray(im.convert("RGB"), np.float32)
    ang, coh = grain_info(im)
    eo = band_energy(im)
    return (f"{label:22s} size={im.width}x{im.height} mean=#{'%02x%02x%02x' % tuple(a.reshape(-1,3).mean(0).astype(int))} "
            f"std={a.std():5.1f} grain={ang:3.0f}deg/{coh:.2f} bands={[round(float(e), 1) for e in eo]} "
            f"seam={seam_ratio(im.convert('RGB')):.2f}")


def compare_view(uid, path=None, cols=40):
    """The side-by-side check, in text (no image decoder in this environment): for luminance and for
    edges, ORIGINAL | RESULT shrunk back to the original's size | RESULT 1:1 centre crop, then a DIFF
    map showing where the shrunk result differs from the original. Print or Read this for every unit."""
    o = orig(uid, "RGBA")
    out = Image.open(path) if path else orig(uid)
    s = out.resize(o.size, Image.LANCZOS)
    w = min(o.size)
    crop = out.crop(((out.width - w) // 2, (out.height - w) // 2, (out.width - w) // 2 + w, (out.height - w) // 2 + w))
    panels = []
    for mode in ("lum", "edge"):
        vs = [text_view(im, cols=cols, mode=mode).split("\n")[1:] for im in (o, s, crop)]
        width = cols + 3
        head = ("ORIGINAL %dx%d" % o.size).ljust(width) + ("RESULT shrunk").ljust(width) + "RESULT 1:1 crop"
        body = ["  ".join((v[i] if i < len(v) else "").ljust(width) for v in vs)
                for i in range(max(len(v) for v in vs))]
        panels.append(f"[{mode}]\n{head}\n" + "\n".join(body))
    def lum(im):
        a = np.asarray(_as_img(im).convert("RGBA"), np.float32) / 255.0
        al = a[..., 3:4]
        r = a[..., :3] * al + 0.35 * (1 - al)
        return 0.299 * r[..., 0] + 0.587 * r[..., 1] + 0.114 * r[..., 2]
    d = np.abs(lum(o) - lum(s))
    dv = np.clip(d / max(np.percentile(d, 98), 1e-6), 0, 1)
    hh, ww = d.shape
    rr = max(2, int(round(cols * hh / ww / 2)))
    ds = cv2.resize(dv, (cols, rr), interpolation=cv2.INTER_AREA)
    dlines = ["".join(_RAMP[c] for c in np.clip((r_ * (len(_RAMP) - 1)).astype(int), 0, len(_RAMP) - 1)) for r_ in ds]
    panels.append("[diff original vs result shrunk: blank = identical, busy = the design moved]\n" + "\n".join(dlines))
    return ("\n".join([stats_view(o, "original"), stats_view(out, "result"), stats_view(s, "result shrunk")])
            + "\n" + "\n\n".join(panels))


def grain_info(img, max_side=256):
    """(streak_deg, coherence) of an image's dominant linear structure (wood grain, streaks, brick
    courses, corrugation). streak_deg: direction the streaks RUN, 0 = left-right, 90 = top-bottom
    (measured clockwise on screen, mod 180). coherence 0..1: ~0 isotropic noise, >0.3 clearly
    directional. Orientation is meaningless when coherence is low."""
    g = np.asarray(img.convert("L"), np.float32)
    if max(g.shape) > max_side:
        k = max_side / max(g.shape)
        g = cv2.resize(g, (max(8, int(g.shape[1] * k)), max(8, int(g.shape[0] * k))), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = float((gx * gx).mean()), float((gy * gy).mean()), float((gx * gy).mean())
    tr = jxx + jyy + 1e-9
    coh = float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / tr)
    grad_deg = np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy)) % 180.0
    return float((grad_deg + 90.0) % 180.0), coh


def angle_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def align_grain(asset, ref, min_coherence=0.15):
    """Rotate `asset` (a big library image) so its dominant grain/streak direction matches `ref`
    (the original, any size), then crop the largest clean square from the middle. Use this BEFORE
    region(): a wood asset whose grain runs sideways must be turned to match a vertical-grain original.
    Returns asset unchanged if `ref` has no clear direction. Prints what it did."""
    a_ref, c_ref = grain_info(ref)
    if c_ref < min_coherence:
        return asset
    a_as, c_as = grain_info(asset)
    delta = (a_ref - a_as + 90.0) % 180.0 - 90.0           # smallest rotation, clockwise
    print(f"align_grain: ref streaks {a_ref:.0f}deg (coh {c_ref:.2f}), asset {a_as:.0f}deg (coh {c_as:.2f}) -> rotate {delta:.0f}deg")
    if abs(delta) < 3:
        return asset
    s = min(asset.size)
    sq = asset.crop(((asset.width - s) // 2, (asset.height - s) // 2, (asset.width + s) // 2, (asset.height + s) // 2))
    r = sq.rotate(-delta, resample=Image.BICUBIC)
    th = np.radians(abs(delta) % 90)
    inner = int(s / (np.cos(th) + np.sin(th))) if (abs(delta) % 90) > 1 else s
    return r.crop(((s - inner) // 2, (s - inner) // 2, (s + inner) // 2, (s + inner) // 2))


def band_energy(img, bands=((0.06, 0.12), (0.12, 0.25), (0.25, 0.5))):
    """Spectral energy of the luminance in cycles/pixel bands (Hann-windowed FFT), normalised so
    flat images give ~0. Used to check that the visible structure scales survive."""
    g = np.asarray(img.convert("L"), np.float32)
    g = g - g.mean()
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = np.abs(np.fft.fftshift(np.fft.fft2(g * win))) ** 2 / (win ** 2).sum()
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    return [float(F[(r >= lo) & (r < hi)].sum()) for lo, hi in bands]


# ---------------------------------------------------------------- metrics shared with gate.py
def seam_ratio(img):
    """How visible the wrap-around seam is: edge-to-edge difference over typical neighbour
    difference. ~1 = tiles seamlessly; >2 = visible seam."""
    a = to_f(img)
    def one(x):
        edge = np.abs(x[:, 0] - x[:, -1]).mean()
        inner = np.abs(np.diff(x, axis=1)).mean() + 1e-4
        return edge / inner
    return float((one(a) + one(a.transpose(1, 0, 2))) / 2)


def soft_alpha(img):
    """Mirror of the engine's AlphaIsSoft: soft when <45% of visible pixels are near-opaque."""
    a = np.asarray(img.convert("RGBA").getchannel("A"))
    ink = int((a >= 32).sum())
    opaque = int((a >= 200).sum())
    return ink == 0 or opaque < ink * 0.45


# ---------------------------------------------------------------- output
OUT = ROOT / "out"


def finish(uid, img, task, method, sources=(), fidelity="exact", notes="", alpha=None):
    """Write the final texture + provenance sidecar to out/<task>/<uid>.png|.json.

    img      : PIL image, any size (resized to the unit's target size if different)
    method   : short string, e.g. "collage+colormatch", "vectorize", "svg-redraw", "freq-inject"
    sources  : library asset ids used (must exist in assets/index.jsonl; all CC0)
    fidelity : "exact"    - design must be preserved (logos, glyphs, gauges, nose art, text, labels);
                            gate demands the result matches the original when shrunk back.
               "material" - only the surface look matters (gravel, brick, metal, wood...); gate checks
                            colour/brightness layout, contrast and tiling instead.
    alpha    : optional PIL 'L' alpha for textures that have one (default: original's, smooth)
    """
    u = info(uid)
    tw, th = u["target"]
    for s in sources:
        lib_rec(s)                                           # raises on unknown ids
    d = OUT / task
    d.mkdir(parents=True, exist_ok=True)
    rgb = img.convert("RGB")
    if rgb.size != (tw, th):
        rgb = rgb.resize((tw, th), Image.LANCZOS)
    if u["has_alpha"]:
        a = alpha if alpha is not None else alpha_from_orig(uid, (tw, th))
        if a.size != (tw, th):
            a = a.resize((tw, th), Image.BICUBIC)
        rgb = rgb.copy()
        rgb.putalpha(a)
    rgb.save(d / f"{uid}.png", optimize=True)
    json.dump({"uid": uid, "status": "ok", "method": method, "fidelity": fidelity,
               "sources": list(sources), "notes": notes}, open(d / f"{uid}.json", "w"), indent=1)
    return d / f"{uid}.png"


def mark(uid, task, status, reason=""):
    """status: 'esrgan_needed' (nothing else can do it - e.g. painted art with no vector/asset route),
    'keep_original' (not worth upgrading, e.g. a plain flat colour), or 'failed'."""
    assert status in ("esrgan_needed", "keep_original", "failed")
    d = OUT / task
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"uid": uid, "status": status, "notes": reason}, open(d / f"{uid}.json", "w"), indent=1)


__all__ = [n for n in dir() if not n.startswith("_")]
