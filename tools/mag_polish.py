"""mag_polish.py: clean-up passes shared by the eight scrapbook magazine covers (all work on the RGBA page at 4x the original, same geometry as the original).
  lock(rgb, ref, sc, sl)   pull the low-frequency colour (Lab ab, sigma sc) and brightness (L, sigma sl) of the redraw back to the original's, so haze, glow and colour casts the redraw invented go away while its fine detail stays
  smooth_alpha(alpha1x, k)   the original's binary alpha upscaled with a smooth contour instead of 4 px stair steps
  ref_page(stem)           the original Lanczos-upscaled 4x over a smooth colour extension (what lock() compares with)
CLI:  mag_polish.py STEM IN.png OUT.png [--sc N] [--sl N] [--mask MASK.png]    (mask: 0..255, where the lock applies, default everywhere)"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tile_mags_rot as tr                                                       # noqa: E402

SB = tr.SB


def ref_page(stem, scale=4):
    im = np.asarray(Image.open(SB / (stem + ".PNG")).convert("RGBA"))
    rgb = tr.extend(im)
    H, W = rgb.shape[:2]
    return cv2.resize(rgb, (W * scale, H * scale), interpolation=cv2.INTER_LANCZOS4)


def smooth_alpha(stem, scale=4, blur=0.9):
    a = np.asarray(Image.open(SB / (stem + ".PNG")).convert("RGBA").getchannel("A")).astype(np.float32) / 255
    H, W = a.shape
    a = cv2.GaussianBlur(a, (0, 0), blur)
    big = cv2.resize(a, (W * scale, H * scale), interpolation=cv2.INTER_CUBIC)
    t = np.clip((big - 0.5) / 0.16 * 0.5 + 0.5, 0, 1)                             # soft step about 0.5, ~1.3 px wide at 4x
    return (t * 255 + .5).astype(np.uint8)


def lock(rgb, ref, sc=28, sl=24, mask=None):
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    lr = cv2.cvtColor(ref.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    out = lab.copy()
    for ch, s in ((0, sl), (1, sc), (2, sc)):
        if s <= 0:
            continue
        d = cv2.GaussianBlur(lr[..., ch], (0, 0), s) - cv2.GaussianBlur(lab[..., ch], (0, 0), s)
        out[..., ch] = lab[..., ch] + (d if mask is None else d * mask)
    res = cv2.cvtColor(out, cv2.COLOR_Lab2RGB)
    return (res.clip(0, 1) * 255 + .5).astype(np.uint8)


def polish(stem, src, dst, sc=28, sl=24, mask=None, alpha=True):
    im = Image.open(src).convert("RGBA")
    rgb = np.asarray(im)[..., :3]
    ref = ref_page(stem, rgb.shape[1] // Image.open(SB / (stem + ".PNG")).width)
    out = lock(rgb, ref, sc, sl, mask)
    a = smooth_alpha(stem, rgb.shape[1] // Image.open(SB / (stem + ".PNG")).width) if alpha else np.asarray(im)[..., 3]
    Image.fromarray(np.dstack([out, a])).save(dst)
    return dst


if __name__ == "__main__":
    a = sys.argv[1:]
    sc, sl, mk = 28, 24, None
    if "--sc" in a: i = a.index("--sc"); sc = float(a[i + 1]); del a[i:i + 2]
    if "--sl" in a: i = a.index("--sl"); sl = float(a[i + 1]); del a[i:i + 2]
    if "--mask" in a: i = a.index("--mask"); mk = np.asarray(Image.open(a[i + 1]).convert("L")).astype(np.float32) / 255; del a[i:i + 2]
    print(polish(a[0], a[1], a[2], sc, sl, mk))


def box_mask(shape, boxes, feather=40):
    """soft 0..1 mask: union of (x0,y0,x1,y1) rectangles in page pixels, edges feathered by `feather`"""
    m = np.zeros(shape[:2], np.float32)
    for x0, y0, x1, y1 in boxes:
        m[y0:y1, x0:x1] = 1
    return cv2.GaussianBlur(m, (0, 0), feather / 2) if feather else m


def local_lock(stem, src, dst, boxes, sc=8, sl=5, feather=40):
    """apply a strong lock (small sigmas: only detail finer than sl px is kept from the redraw) inside the boxes only"""
    im = Image.open(src).convert("RGBA")
    arr = np.asarray(im)
    k = arr.shape[1] // Image.open(SB / (stem + ".PNG")).width
    m = box_mask(arr.shape, boxes, feather)
    out = lock(arr[..., :3], ref_page(stem, k), sc, sl, m)
    Image.fromarray(np.dstack([out, arr[..., 3]])).save(dst)
    return dst


def sharp(rgb, s=2.5, amt=1.0):
    f = rgb.astype(np.float32)
    return (f + amt * (f - cv2.GaussianBlur(f, (0, 0), s))).clip(0, 255)


def ghost_mask(klein, ref, boxes, sig=8, lo=5, hi=14, feather=24, bfeather=30):
    """where the redraw's low-frequency brightness disagrees with the original's (ghost shapes), inside the boxes"""
    L1 = cv2.cvtColor(klein, cv2.COLOR_RGB2GRAY).astype(np.float32)
    L0 = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32)
    d = np.abs(cv2.GaussianBlur(L1, (0, 0), sig) - cv2.GaussianBlur(L0, (0, 0), sig))
    m = np.clip((d - lo) / (hi - lo), 0, 1)
    m = cv2.dilate(m, np.ones((9, 9), np.uint8))
    m = cv2.GaussianBlur(m, (0, 0), feather / 2) * box_mask(klein.shape, boxes, bfeather)
    return m


def red_mask(rgb, grow=3):
    f = rgb.astype(np.int16)
    m = ((f[..., 0] - f[..., 1] > 60) & (f[..., 0] > 120)).astype(np.uint8)
    return cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))


def gold_mask(rgb, grow=4):
    f = rgb.astype(np.int16)
    m = ((f[..., 0] > 170) & (f[..., 1] > 120) & (f[..., 2] < 110) & (f[..., 0] - f[..., 2] > 90)).astype(np.uint8)
    return cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))


def ghost_fix(stem, src, dst, boxes, sig=8, lo=5, hi=14, s=2.5, amt=1.0, grain=2.0, keep_red=False, keep_gold=False, bfeather=30):
    """keep_red: the crisp red lettering of the redraw stays; the original's own (blurred) red lettering is painted out of the reference first"""
    im = Image.open(src).convert("RGBA"); arr = np.asarray(im)
    k = arr.shape[1] // Image.open(SB / (stem + ".PNG")).width
    ref = ref_page(stem, k)
    if keep_red:
        ref = cv2.inpaint(ref, red_mask(ref, 5) * 255, 7, cv2.INPAINT_TELEA)
    m = ghost_mask(arr[..., :3], ref, boxes, sig, lo, hi, bfeather=bfeather)
    if keep_red:
        m = m * (1 - cv2.GaussianBlur(red_mask(arr[..., :3], 4).astype(np.float32), (0, 0), 3))
    if keep_gold:
        m = m * (1 - cv2.GaussianBlur(gold_mask(arr[..., :3], 4).astype(np.float32), (0, 0), 3))
    o = sharp(ref, s, amt)
    if grain:
        o = o + np.random.default_rng(1).normal(0, grain, o.shape[:2])[..., None]
    out = (arr[..., :3] * (1 - m[..., None]) + o * m[..., None]).clip(0, 255).astype(np.uint8)
    Image.fromarray(np.dstack([out, arr[..., 3]])).save(dst)
    Image.fromarray((m * 255).astype(np.uint8)).save(str(dst) + ".mask.png")
    return dst


def motion_blur(rgb, length=21, angle=0.0):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    M = cv2.getRotationMatrix2D((length / 2 - .5, length / 2 - .5), angle, 1.0)
    k = cv2.warpAffine(k, M, (length, length), flags=cv2.INTER_LINEAR)
    return cv2.filter2D(rgb.astype(np.float32), -1, k / k.sum())


def blur_in(src, dst, boxes, sigma=2.0, length=0, angle=0.0, feather=30, keep_gold=True):
    """soften a hard-edged hallucination: gaussian (or directional motion) blur inside the boxes, lettering kept"""
    arr = np.asarray(Image.open(src).convert("RGBA"))
    rgb = arr[..., :3]
    b = motion_blur(rgb, length, angle) if length else cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), sigma)
    m = box_mask(arr.shape, boxes, feather)
    if keep_gold:
        m = m * (1 - cv2.GaussianBlur(gold_mask(rgb, 4).astype(np.float32), (0, 0), 2))
    out = rgb * (1 - m[..., None]) + b * m[..., None]
    Image.fromarray(np.dstack([out.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)


def replace_from_ref(stem, src, dst, boxes, s=2.0, amt=1.1, feather=14, grain=1.5):
    """paint the boxes from the original (Lanczos 4x, lightly sharpened) where the redraw invented detail that is not in the picture (small icons)"""
    arr = np.asarray(Image.open(src).convert("RGBA"))
    k = arr.shape[1] // Image.open(SB / (stem + ".PNG")).width
    o = sharp(ref_page(stem, k), s, amt) + np.random.default_rng(2).normal(0, grain, arr.shape[:2])[..., None]
    m = box_mask(arr.shape, boxes, feather)[..., None]
    out = arr[..., :3] * (1 - m) + o * m
    Image.fromarray(np.dstack([out.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)


def fill_poly(src, dst, poly, paper, sig=(4, 12, 40), grain=1.6):
    """repaint a polygon with the paper colour around it (normalised convolution over the pixels `paper(rgb)` accepts outside the polygon)"""
    arr = np.asarray(Image.open(src).convert("RGBA")); rgb = arr[..., :3].astype(np.float32)
    pm = np.zeros(arr.shape[:2], np.uint8); cv2.fillPoly(pm, [np.array(poly, np.int32)], 1)
    keep = (paper(arr[..., :3]) & (cv2.dilate(pm, np.ones((9, 9), np.uint8)) == 0)).astype(np.float32)
    out, filled = rgb * keep[..., None], keep.copy()
    for s in sig:
        num = cv2.GaussianBlur(rgb * keep[..., None], (0, 0), s); den = cv2.GaussianBlur(keep, (0, 0), s)
        new = (filled < .5) & (den > .02)
        out[new] = (num / np.maximum(den, 1e-6)[..., None])[new]; filled[new] = 1
    out += np.random.default_rng(3).normal(0, grain, arr.shape[:2])[..., None]
    m = cv2.GaussianBlur(pm.astype(np.float32), (0, 0), 1.2)[..., None]
    res = rgb * (1 - m) + out * m
    Image.fromarray(np.dstack([res.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)


WIN = Path.home() / ".local/share/fonts/winfonts"


def tilted_text(src, dst, text, x0, base0, x1, slope, cap, font, ink, paper, margin=5, seed=5, weight=1.0, shadow=0.0, rect_up=1.5, rect_down=0.5, thin_th=30, wipe=True, soft=0.5, stroke=0, space=2.4):
    """repaint a short tilted label: the old lettering is wiped with the paper colour around it (normalised convolution over `paper`), then `text` is set in `font` along the line through
    (x0, base0) with the given slope (dy/dx) so that it spans x0..x1 with capitals `cap` px high (letter-spaced to fit), in colour `ink`"""
    from PIL import ImageDraw, ImageFont
    arr = np.asarray(Image.open(src).convert("RGBA")); H, W = arr.shape[:2]
    ang = np.degrees(np.arctan(slope)); L = (x1 - x0) / np.cos(np.arctan(slope))
    f0 = ImageFont.truetype(str(font), 200); hb = f0.getbbox("H"); size = 200 * cap / (hb[3] - hb[1]); f = ImageFont.truetype(str(font), int(round(size)))
    widths = [f.getlength(c) * (space if c == ' ' else 1.0) for c in text]; gap = (L - sum(widths)) / max(len(text) - 1, 1)
    lay = Image.new("L", (int(L) + 40, int(cap * 3)), 0); d = ImageDraw.Draw(lay); x = 20.0
    for c, w in zip(text, widths):
        d.text((x, cap * 2), c, font=f, fill=255, anchor="ls", stroke_width=stroke, stroke_fill=255); x += w + gap
    if weight != 1.0:
        lay = Image.fromarray(cv2.GaussianBlur(np.asarray(lay).astype(np.float32), (0, 0), 0.6).clip(0, 255).astype(np.uint8))
    M = cv2.getRotationMatrix2D((20, cap * 2), -ang, 1.0)
    M[0, 2] += x0 - 20; M[1, 2] += base0 - cap * 2
    a = cv2.warpAffine(np.asarray(lay), M, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
    rect = Image.new("L", lay.size, 0); ImageDraw.Draw(rect).rectangle([0, int(cap * 2 - cap * rect_up), lay.size[0], int(cap * 2 + cap * rect_down)], fill=255)
    rect = cv2.warpAffine(np.asarray(rect), M, (W, H), flags=cv2.INTER_NEAREST) > 128
    wipe_m = (cv2.dilate((a > 0.02).astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) > 0)
    if paper is None:                                                                                                # lettering on a photo: its thin bright strokes are the old ink
        Lh = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
        th = Lh - cv2.morphologyEx(Lh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))
        wipe_m = cv2.dilate(((th > thin_th) & rect & (arr[..., 2] > 165)).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0     # cream ink only: skin and metal highlights stay
        paper = lambda rgb_: np.ones(rgb_.shape[:2], bool)
    else:
        wipe_m |= rect & (cv2.dilate((~paper(arr[..., :3])).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)    # every dark speck of the old label text inside the line's box
    rgb = arr[..., :3].astype(np.float32)
    keep = (paper(arr[..., :3]) & ~(cv2.dilate(wipe_m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)).astype(np.float32)
    out, filled = rgb * keep[..., None], keep.copy()
    for s in (3, 8, 24):
        num = cv2.GaussianBlur(rgb * keep[..., None], (0, 0), s); den = cv2.GaussianBlur(keep, (0, 0), s)
        new = (filled < .5) & (den > .02)
        out[new] = (num / np.maximum(den, 1e-6)[..., None])[new]; filled[new] = 1
    out += np.random.default_rng(seed).normal(0, 1.6, (H, W))[..., None]
    wm = cv2.GaussianBlur(wipe_m.astype(np.float32), (0, 0), 1.0)[..., None] * (1.0 if wipe else 0.0)
    res = rgb * (1 - wm) + out * wm
    al = cv2.GaussianBlur(a, (0, 0), soft)[..., None]
    if shadow:
        sh = cv2.GaussianBlur(np.roll(np.roll(a, 3, 0), 2, 1), (0, 0), 1.6)[..., None] * shadow
        res = res * (1 - sh) + np.array((15, 10, 8), np.float32) * sh
    res = res * (1 - al) + np.array(ink, np.float32) * al
    Image.fromarray(np.dstack([res.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)


def band_poly(x0, x1, y0, y1, half):
    """polygon: the band of +-half px around the straight line from (x0, y0) to (x1, y1)"""
    return [(x0, y0 - half), (x1, y1 - half), (x1, y1 + half), (x0, y0 + half)]


def bitmap_text(stem, src, dst, poly, ink=(238, 229, 204), shadow=0.5, thr=(0.35, 0.65), wipe_th=35, open_k=23, min_diag=110, join=28, pre=0.55):
    """repaint lettering from the original's own pixels: the thin bright strokes of the original (top-hat of its brightness) are upscaled with Lanczos, their edges steepened and laid
    over the redraw after the redraw's garbled strokes were wiped; right letters, the original's shapes, crisp edges. Inside the polygon only."""
    arr = np.asarray(Image.open(src).convert("RGBA")); H, W = arr.shape[:2]; rgb = arr[..., :3].astype(np.float32)
    k = W // Image.open(SB / (stem + ".PNG")).width
    pm = np.zeros((H, W), np.uint8); cv2.fillPoly(pm, [np.array(poly, np.int32)], 1)
    Lh = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    th = Lh - cv2.morphologyEx(Lh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    wipe_m = cv2.dilate(((th > wipe_th) & (pm > 0)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    keep = ((~wipe_m) & (cv2.dilate(pm, np.ones((15, 15), np.uint8)) > 0)).astype(np.float32)
    out, filled = rgb * keep[..., None], keep.copy()
    for s in (3, 8, 24, 60):
        num = cv2.GaussianBlur(rgb * keep[..., None], (0, 0), s); den = cv2.GaussianBlur(keep, (0, 0), s)
        new = (filled < .5) & (den > .02)
        out[new] = (num / np.maximum(den, 1e-6)[..., None])[new]; filled[new] = 1
    out += np.random.default_rng(7).normal(0, 1.6, (H, W))[..., None]
    wm = cv2.GaussianBlur(wipe_m.astype(np.float32), (0, 0), 1.2)[..., None]
    res = rgb * (1 - wm) + out * wm
    o1 = np.asarray(Image.open(SB / (stem + ".PNG")).convert("RGB"))[..., 2].astype(np.float32)               # blue channel: the cream lettering is bright in it, skin highlights and metal are not
    t1 = o1 - cv2.morphologyEx(o1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    pm1 = cv2.resize(pm, (W // k, H // k), interpolation=cv2.INTER_NEAREST) > 0
    scale = np.percentile(t1[pm1], 97) if pm1.any() else 100
    c = cv2.GaussianBlur(np.clip(t1 / max(scale, 1), 0, 1), (0, 0), pre)
    c4 = cv2.resize(c, (W, H), interpolation=cv2.INTER_LANCZOS4).clip(0, 1)
    c4 = np.clip((c4 - thr[0]) / (thr[1] - thr[0]), 0, 1); c4 = c4 * c4 * (3 - 2 * c4)
    seeds = c4 > 0.5                                                              # keep letter strokes, drop the specks of hands, ring and buckle: components that are long, plus the small ones (dots, bars) next to them
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(seeds.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))), connectivity=8)
    big = np.zeros(seeds.shape, np.uint8)
    for i in range(1, n):
        if np.hypot(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) >= min_diag:
            big[(lab == i) & seeds] = 1
    near = cv2.dilate(big, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * join + 1, 2 * join + 1))) > 0
    c4 = c4 * cv2.GaussianBlur(near.astype(np.float32), (0, 0), 1.5)
    c4 = c4 * cv2.GaussianBlur(pm.astype(np.float32), (0, 0), 2.0)
    sh = cv2.GaussianBlur(np.roll(np.roll(c4, 3, 0), 2, 1), (0, 0), 1.6) * shadow
    res = res * (1 - sh[..., None]) + np.array((15, 10, 8), np.float32) * sh[..., None]
    res = res * (1 - c4[..., None]) + np.array(ink, np.float32) * c4[..., None]
    Image.fromarray(np.dstack([res.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)


# ---- second-pass helpers (visual QA pass 2) ----

def _fill(rgb, keep, todo, sig=(3, 8, 24, 60)):
    """normalised convolution: every `todo` pixel gets the local mean of the `keep` pixels around it (growing sigma until something is found)"""
    rgb = rgb.astype(np.float32); keep = keep.astype(np.float32)
    out = rgb.copy(); filled = (~todo).astype(np.float32) if todo.dtype == bool else 1 - todo
    num0 = rgb * keep[..., None]
    for s in sig:
        num = cv2.GaussianBlur(num0, (0, 0), s); den = cv2.GaussianBlur(keep, (0, 0), s)
        new = (filled < .5) & (den > .02)
        out[new] = (num / np.maximum(den, 1e-6)[..., None])[new]; filled[new] = 1
    return out


def poly_mask(shape, polys):
    m = np.zeros(shape[:2], np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.array(p, np.int32)], 1)
    return m > 0


def text_mask(rgb, th=30, open_k=15, minL=150, grow=2, bright=True):
    """crisp lettering: thin strokes brighter (or darker) than their surroundings"""
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    if bright:
        m = ((L - cv2.morphologyEx(L, cv2.MORPH_OPEN, se)) > th) & (L > minL)
    else:
        m = ((cv2.morphologyEx(L, cv2.MORPH_CLOSE, se) - L) > th) & (L < minL)
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0 if grow else m


def clean_bg(src, dst, polys, protect, dev=7, med=31, sig=(4, 10, 30, 80), grain=1.5, all_bg=False, seed=11, feather=1.2, thin=0, median=0):
    """inside the polygons, everything that is not `protect` (a bool mask, e.g. the crisp lettering) and deviates from its local median by more than `dev` (ghost strokes, specks,
    scratches) is repainted with the surrounding background (all_bg: every unprotected pixel is repainted, the texture there goes flat + grain)"""
    arr = np.asarray(Image.open(src).convert("RGBA")); rgb = arr[..., :3]
    pm = poly_mask(arr.shape, polys)
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if thin:                                                                     # only thin strokes (ghost letters, scratches): broad tone edges stay
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thin, thin)); Lf = L.astype(np.float32)
        odd = np.maximum(Lf - cv2.morphologyEx(Lf, cv2.MORPH_OPEN, se), cv2.morphologyEx(Lf, cv2.MORPH_CLOSE, se) - Lf) > dev
    else:
        Lm = cv2.medianBlur(L, med).astype(np.float32)
        odd = np.abs(L.astype(np.float32) - Lm) > dev
    odd = cv2.dilate(odd.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    todo = pm & ~protect & (odd | all_bg)
    keep = ~protect & ~todo & (arr[..., 3] > 250)
    if median:                                                                   # robust local median of the unprotected pixels (keeps tone edges, no blotches from sparse keep pixels)
        base = _fill(rgb, ~protect, protect, sig).clip(0, 255).astype(np.uint8)
        out = np.dstack([cv2.medianBlur(base[..., c], median) for c in range(3)]).astype(np.float32)
    else:
        out = _fill(rgb, keep, todo, sig)
    out = out + np.random.default_rng(seed).normal(0, grain, L.shape)[..., None]
    w = cv2.GaussianBlur(todo.astype(np.float32), (0, 0), feather)[..., None] if feather else todo[..., None].astype(np.float32)
    w = w * (~protect)[..., None]
    res = rgb * (1 - w) + out * w
    Image.fromarray(np.dstack([res.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)
    return dst


def ref_in_mask(stem, src, dst, mask, s=2.0, amt=1.1, grain=1.5, feather=6, match=False):
    """like replace_from_ref, but over an arbitrary 0..1 mask (soft edges by `feather`); match: the original's patch is first given the redraw's mean colour there"""
    arr = np.asarray(Image.open(src).convert("RGBA"))
    k = arr.shape[1] // Image.open(SB / (stem + ".PNG")).width
    o = sharp(ref_page(stem, k), s, amt) + np.random.default_rng(2).normal(0, grain, arr.shape[:2])[..., None]
    m = mask.astype(np.float32)
    if feather:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    m = m[..., None]
    out = arr[..., :3] * (1 - m) + o * m
    Image.fromarray(np.dstack([out.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)
    return dst


def edge_band(alpha, width, polys=None):
    """pixels of the page within `width` px of its transparent outside, limited to the polygons"""
    inside = (alpha > 127).astype(np.uint8)
    d = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    m = (d > 0) & (d < width)
    if polys is not None:
        m &= poly_mask(alpha.shape, polys)
    return m


def page_layer(stem, a_up, scale=4):
    """a 0..1 layer drawn in the straightened 4x frame of tile_mags_rot (the frame the mag_sb_* scripts typeset in) moved into the page frame"""
    im = Image.open(SB / (stem + ".PNG")); W, H = im.size
    M, _ = tr.forward(W, H, tr.ANGLE[stem])
    Sm = np.diag([float(scale), float(scale), 1.0])
    M4 = (Sm @ np.vstack([M, [0, 0, 1]]) @ np.linalg.inv(Sm))[:2]
    return cv2.warpAffine(a_up.astype(np.float32), M4, (W * scale, H * scale), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


def upright_shape(stem, scale=4):
    im = Image.open(SB / (stem + ".PNG")); W, H = im.size
    _, (W2, H2) = tr.forward(W, H, tr.ANGLE[stem])
    return H2 * scale, W2 * scale


def wipe_and_set(src, dst, wipe_m, a, ink, sig=(2, 5, 12, 30), grain=1.6, seed=13, shadow=0.0, sh_off=(2, 2), sh_blur=1.4, sh_col=(15, 10, 8), keep=None):
    """old ink (wipe_m) refilled from its unmasked neighbourhood, then the layer `a` (0..1, page frame) composited in `ink`"""
    arr = np.asarray(Image.open(src).convert("RGBA")); rgb = arr[..., :3].astype(np.float32)
    kp = ~wipe_m if keep is None else (keep & ~wipe_m)
    out = _fill(rgb, kp, wipe_m, sig) + np.random.default_rng(seed).normal(0, grain, wipe_m.shape)[..., None]
    w = cv2.GaussianBlur(wipe_m.astype(np.float32), (0, 0), 0.8)[..., None]
    res = rgb * (1 - w) + out * w
    if shadow:
        sh = cv2.GaussianBlur(np.roll(np.roll(a, sh_off[0], 0), sh_off[1], 1), (0, 0), sh_blur)[..., None] * shadow
        res = res * (1 - sh) + np.array(sh_col, np.float32) * sh
    res = res * (1 - a[..., None]) + np.array(ink, np.float32) * a[..., None]
    Image.fromarray(np.dstack([res.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)
    return dst


def ref_clean(stem, scale=4, red_grow=5, inpaint=None):
    """the original 4x with its (blurred) red lettering painted out, for areas where new crisp red lettering is laid over the original's pixels"""
    ref = ref_page(stem, scale)
    f = ref.astype(np.int16)                                                     # red, not purple (purple has a high red too)
    m = ((f[..., 0] - f[..., 1] > 60) & (f[..., 0] > 120) & (f[..., 0] - f[..., 2] > 60)).astype(np.uint8)
    m = cv2.dilate(m, np.ones((2 * red_grow + 1, 2 * red_grow + 1), np.uint8)) * 255
    if inpaint is not None:
        m = m * inpaint.astype(np.uint8)
    return cv2.inpaint(ref, m.astype(np.uint8), 9, cv2.INPAINT_TELEA)


def blend_rgb(src, dst, img, mask, feather=0):
    arr = np.asarray(Image.open(src).convert("RGBA"))
    m = mask.astype(np.float32)
    if feather:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    m = m[..., None]
    out = arr[..., :3] * (1 - m) + img.astype(np.float32) * m
    Image.fromarray(np.dstack([out.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)
    return dst


def thin_out(src, dst, mask, k=9, th=18, med=15, seed=5):
    """thin bright lines (scratches, crackle) inside `mask` toned down to their surroundings (morphological opening / median)"""
    arr = np.asarray(Image.open(src).convert("RGBA")); rgb = arr[..., :3]
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    odd = ((L - cv2.morphologyEx(L, cv2.MORPH_OPEN, se)) > th) & mask
    odd = cv2.dilate(odd.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    base = np.dstack([cv2.medianBlur(rgb[..., c], med) for c in range(3)]).astype(np.float32)
    op = np.dstack([cv2.morphologyEx(rgb[..., c], cv2.MORPH_OPEN, se) for c in range(3)]).astype(np.float32)
    out = np.minimum(base, op + 6) + np.random.default_rng(seed).normal(0, 1.5, L.shape)[..., None]
    w = cv2.GaussianBlur(odd.astype(np.float32), (0, 0), 1.2)[..., None]
    res = rgb * (1 - w) + out * w
    Image.fromarray(np.dstack([res.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)
    return dst


def set_lines(stem, lines, font_path, x, base, pitch, w0=None, size=None, rot=None, blur=0.5, gain=1.15):
    """alpha (page frame) of `lines` typeset in the straightened 4x frame the mag_sb_* scripts use: left x, first baseline, line pitch; the font size makes line 0 span w0 px
    (or is `size`); rot=(cx, cy, ang): the text block is turned like the scripts' tilted captions"""
    from PIL import ImageDraw, ImageFont
    Hu, Wu = upright_shape(stem)
    if size is None:
        f = ImageFont.truetype(str(font_path), 100); size = 100 * w0 / f.getlength(lines[0])
    f = ImageFont.truetype(str(font_path), size)
    im = Image.new("L", (Wu, Hu), 0); d = ImageDraw.Draw(im)
    for k, t in enumerate(lines):
        d.text((x, base + k * pitch), t, font=f, fill=255, anchor="ls")
    a = np.clip(cv2.GaussianBlur(np.asarray(im, np.float32) / 255, (0, 0), blur) * gain, 0, 1) if blur else np.asarray(im, np.float32) / 255
    if rot is not None:
        M = cv2.getRotationMatrix2D(rot[:2], rot[2], 1.0)
        a = cv2.warpAffine(a, M, (Wu, Hu), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return page_layer(stem, a)


def ref_notext(stem, scale=4, k=15, th=18, grow=7, region=None, red_grow=0):
    """the original 4x with its own (blurred) light lettering painted out (top-hat of its brightness, inpainted): a background to lay the redraw's crisp lettering on"""
    ref = ref_page(stem, scale)
    L = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = (L - cv2.morphologyEx(L, cv2.MORPH_OPEN, se)) > th
    if red_grow:
        m |= red_mask(ref, 2) > 0
    m = cv2.dilate(m.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    if region is not None:
        m = m * region.astype(np.uint8)
    return cv2.inpaint(ref, m * 255, 9, cv2.INPAINT_TELEA)


def redo_edge(src, dst, xs, ys, side_a, side_b, half=10, protect=None, x_range=None, seed=17):
    """a colour boundary the redraw made ragged (steps, bites) is redrawn as the smooth curve through (xs, ys) (cubic fit): within +-half px of it every unprotected pixel
    is refilled from the clean pixels of its own side (side_a above the curve, side_b below; both are bool masks of the colours the sides should have)"""
    arr = np.asarray(Image.open(src).convert("RGBA")); rgb = arr[..., :3].astype(np.float32); H, W = arr.shape[:2]
    p = np.polyfit(xs, ys, 3)
    yy, xx = np.mgrid[0:H, 0:W]
    yc = np.polyval(p, xx)
    d = yy - yc
    band = np.abs(d) < half
    if x_range is not None:
        band &= (xx >= x_range[0]) & (xx < x_range[1])
    prot = np.zeros((H, W), bool) if protect is None else protect
    todo = band & ~prot
    up, dn = d < 0, d >= 0
    outA = _fill(rgb, side_a & up & ~band & ~prot, np.ones((H, W), bool), (3, 8, 20, 50))
    outB = _fill(rgb, side_b & dn & ~band & ~prot, np.ones((H, W), bool), (3, 8, 20, 50))
    t = np.clip(d + 0.5, 0, 1)[..., None]                                          # anti-aliased 1 px transition along the curve
    out = outA * (1 - t) + outB * t + np.random.default_rng(seed).normal(0, 1.5, (H, W))[..., None]
    w = cv2.GaussianBlur(todo.astype(np.float32), (0, 0), 1.0)[..., None] * (~prot)[..., None]
    res = rgb * (1 - w) + out * w
    Image.fromarray(np.dstack([res.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)
    return dst


def redo_circle(src, dst, cx, cy, r, side_out, side_in, half=9, protect=None, region=None, seed=19):
    """redo_edge for a circular boundary (centre cx, cy, radius r): side_out outside the circle, side_in inside"""
    arr = np.asarray(Image.open(src).convert("RGBA")); rgb = arr[..., :3].astype(np.float32); H, W = arr.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot(xx - cx, yy - cy) - r                                              # >0 outside
    band = np.abs(d) < half
    if region is not None:
        band &= region
    prot = np.zeros((H, W), bool) if protect is None else protect
    todo = band & ~prot
    ok = cv2.dilate(band.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    outA = _fill(rgb, side_out & (d > 0) & ~band & ~prot & ok, np.ones((H, W), bool), (3, 8, 20, 50))
    outB = _fill(rgb, side_in & (d <= 0) & ~band & ~prot & ok, np.ones((H, W), bool), (3, 8, 20, 50))
    t = np.clip(0.5 - d, 0, 1)[..., None]                                           # 1 inside
    out = outA * (1 - t) + outB * t + np.random.default_rng(seed).normal(0, 1.5, (H, W))[..., None]
    w = cv2.GaussianBlur(todo.astype(np.float32), (0, 0), 1.0)[..., None] * (~prot)[..., None]
    res = rgb * (1 - w) + out * w
    Image.fromarray(np.dstack([res.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)
    return dst


def lift_text(rgb, region, ink, th=30, k=13, grow=2, minL=130):
    """split light lettering off its background: returns (background with the letters filled in from around them, letter alpha 0..1). Recomposite with
    bg * (1 - a) + ink * a after the background was repaired."""
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    t = ((L - cv2.morphologyEx(L, cv2.MORPH_OPEN, se)) > th) & region & (L > minL)
    tm = cv2.dilate(t.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
    bg = _fill(rgb.astype(np.float32), ~tm, tm, (2, 4, 8, 20))
    Lb = cv2.cvtColor(bg.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    Li = float(np.dot(ink, (0.299, 0.587, 0.114)))
    a = np.clip((L - Lb) / np.maximum(Li - Lb, 20), 0, 1) * tm
    return bg, a


def redo_circle_text(src, dst, cx, cy, r, side_out, side_in, region, ink, half=10, protect=None, th=30, minL=170):
    """redo_circle under light lettering: the letters are lifted off, the edge redrawn, the letters put back on top"""
    arr = np.asarray(Image.open(src).convert("RGBA")); rgb = arr[..., :3]
    zone = cv2.dilate(region.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    bg, a = lift_text(rgb, zone & (np.zeros(zone.shape, bool) if protect is None else ~(cv2.dilate(protect.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)), ink, th=th, minL=minL)
    tmp_p = str(dst) + ".bg.png"
    Image.fromarray(np.dstack([bg.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(tmp_p)
    bgi = bg.clip(0, 255).astype(np.uint8).astype(int)
    so, si = side_out(bgi), side_in(bgi)
    redo_circle(tmp_p, tmp_p, cx, cy, r, so, si, half=half, protect=protect, region=region)
    nb = np.asarray(Image.open(tmp_p))[..., :3].astype(np.float32)
    res = nb * (1 - a[..., None]) + np.array(ink, np.float32) * a[..., None]
    res = np.where(zone[..., None], res, rgb)
    Image.fromarray(np.dstack([res.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)
    return dst


def fill_under_text(src, dst, polys, paper, ink, th=30, minL=170, sig=(3, 8, 20, 50), seed=23):
    """fill_poly under light lettering: the letters are lifted off, the polygons refilled from the `paper` pixels around them, the letters put back"""
    arr = np.asarray(Image.open(src).convert("RGBA")); rgb = arr[..., :3]
    pm = polys if isinstance(polys, np.ndarray) else poly_mask(arr.shape, polys)            # polygons, or a bool mask
    zone = cv2.dilate(pm.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    bg, a = lift_text(rgb, zone, ink, th=th, minL=minL)
    bgi = bg.clip(0, 255).astype(np.uint8)
    ring = cv2.dilate((a > 0.02).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0                    # the letters' bloom is not paper
    keep = paper(bgi.astype(int)) & ~pm & ~ring
    out = _fill(bg, keep, pm, sig) + np.random.default_rng(seed).normal(0, 1.5, pm.shape)[..., None]
    w = cv2.GaussianBlur(pm.astype(np.float32), (0, 0), 1.5)[..., None]
    nb = bg * (1 - w) + out * w
    res = nb * (1 - a[..., None]) + np.array(ink, np.float32) * a[..., None]
    res = np.where(zone[..., None], res, rgb)
    Image.fromarray(np.dstack([res.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)
    return dst


def circle_side(shape, cx, cy, r, outside=True, margin=1.5):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    d = np.hypot(xx - cx, yy - cy) - r
    return d > margin if outside else d < -margin


def relay_text(src, dst, a, ink, zone, grow=4, fix=None, seed=29):
    """lettering whose exact layer `a` (page frame, 0..1, e.g. re-rendered with set_lines / the mag_sb_* layout) is known: inside `zone` the old letters and their bloom are
    wiped, fix(path_in, path_out) may then repair the bare background (an edge, a smear), and the layer is laid on again in `ink`"""
    arr = np.asarray(Image.open(src).convert("RGBA")); rgb = arr[..., :3].astype(np.float32)
    wm = (cv2.dilate((a > 0.02).astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0) & zone
    out = _fill(rgb, ~wm, wm, (2, 4, 8, 20, 50)) + np.random.default_rng(seed).normal(0, 1.5, wm.shape)[..., None]
    bare = np.where(wm[..., None], out, rgb)
    p = str(dst) + ".bare.png"
    Image.fromarray(np.dstack([bare.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(p)
    if fix is not None:
        fix(p, p)
    nb = np.asarray(Image.open(p))[..., :3].astype(np.float32)
    az = (a * zone)[..., None]
    res = nb * (1 - az) + np.array(ink, np.float32) * az
    res = np.where(zone[..., None], res, rgb)
    Image.fromarray(np.dstack([res.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)
    return dst


def column_layer(stem, lines, para_start, x0, indent, y0, pitch, size, font_r, font_i, blur=0.5, gain=1.15):
    """the body-column layer exactly as the mag_sb_* scripts draw it (<I>..<i> italics), in the page frame"""
    import re
    from PIL import ImageDraw, ImageFont
    Hu, Wu = upright_shape(stem)
    im = Image.new("L", (Wu, Hu), 0); d = ImageDraw.Draw(im)
    fr, fi = ImageFont.truetype(str(font_r), size), ImageFont.truetype(str(font_i), size)
    for i, line in enumerate(lines):
        x = x0 + (indent if i in para_start else 0); yb = y0 + pitch * i + 4.5; it = False
        for seg in re.split(r"(<I>|<i>)", line):
            if seg in ("<I>", "<i>"):
                it = seg == "<I>"; continue
            if seg:
                f = fi if it else fr
                d.text((x, yb), seg, font=f, fill=255, anchor="ls"); x += f.getlength(seg)
    a = np.clip(cv2.GaussianBlur(np.asarray(im, np.float32) / 255, (0, 0), blur) * gain, 0, 1)
    return page_layer(stem, a)


def _M4(stem, scale=4):
    im = Image.open(SB / (stem + ".PNG")); W, H = im.size
    M, (W2, H2) = tr.forward(W, H, tr.ANGLE[stem])
    Sm = np.diag([float(scale), float(scale), 1.0])
    return (Sm @ np.vstack([M, [0, 0, 1]]) @ np.linalg.inv(Sm))[:2], (W2 * scale, H2 * scale), (W * scale, H * scale)


def to_upright(stem, img, scale=4, interp=cv2.INTER_CUBIC):
    """page frame -> the straightened 4x frame of tile_mags_rot (where the mag_sb_* scripts measured their boxes)"""
    M4, (Wu, Hu), _ = _M4(stem, scale)
    return cv2.warpAffine(img, M4, (Wu, Hu), flags=interp, borderMode=cv2.BORDER_REPLICATE)


def to_page(stem, img, scale=4, interp=cv2.INTER_CUBIC):
    M4, _, (W, H) = _M4(stem, scale)
    return cv2.warpAffine(img, M4, (W, H), flags=cv2.WARP_INVERSE_MAP | interp, borderMode=cv2.BORDER_REPLICATE)


def upright_edit(stem, src, dst, fn, scale=4):
    """run fn(upright_rgb_float) -> (new_rgb_float, mask 0..1) in the straightened frame; the result goes back into the page only where mask > 0"""
    arr = np.asarray(Image.open(src).convert("RGBA")); rgb = arr[..., :3].astype(np.float32)
    up = to_upright(stem, rgb, scale)
    new, m = fn(up)
    back = to_page(stem, new.astype(np.float32), scale)
    mb = np.clip(to_page(stem, m.astype(np.float32), scale, cv2.INTER_LINEAR), 0, 1)[..., None]
    res = rgb * (1 - mb) + back * mb
    Image.fromarray(np.dstack([res.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)
    return dst


def colour_text_from_ref(stem, src, dst, poly, measure, ink, thr=(0.35, 0.65), wipe_measure=None, wipe_th=0.25, grow=3, pre=0.0, sig=(2, 5, 12, 30), seed=37):
    """re-extract flat coloured lettering (e.g. the orange headline) from the original's own pixels: measure(rgb_int) -> 0..1 'how much ink' on the 1x original, upscaled
    with Lanczos and its edge steepened (smoothstep between thr) -> the letter alpha at 4x with the original's shapes. The redraw's lettering of that colour (its extra
    outlines, extrusions and glow included: wipe_measure on the page > wipe_th, grown) is refilled from the photo round it first; inside the polygon only."""
    arr = np.asarray(Image.open(src).convert("RGBA")); H, W = arr.shape[:2]; rgb = arr[..., :3].astype(np.float32)
    o = np.asarray(Image.open(SB / (stem + ".PNG")).convert("RGB")).astype(int)
    k = W // o.shape[1]
    pm = poly_mask((H, W), [poly])
    c = measure(o).astype(np.float32)
    if pre:
        c = cv2.GaussianBlur(c, (0, 0), pre)
    c4 = cv2.resize(c, (W, H), interpolation=cv2.INTER_LANCZOS4).clip(0, 1)
    c4 = np.clip((c4 - thr[0]) / (thr[1] - thr[0]), 0, 1); c4 = c4 * c4 * (3 - 2 * c4)
    c4 = c4 * cv2.GaussianBlur(pm.astype(np.float32), (0, 0), 2)
    wm = ((wipe_measure or measure)(arr[..., :3].astype(int)) > wipe_th) & pm
    wm = cv2.dilate(wm.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
    wm |= (c4 > 0.02) & pm
    fill = _fill(rgb, ~wm, wm, sig) + np.random.default_rng(seed).normal(0, 1.6, (H, W))[..., None]
    w = cv2.GaussianBlur(wm.astype(np.float32), (0, 0), 1.0)[..., None]
    res = rgb * (1 - w) + fill * w
    res = res * (1 - c4[..., None]) + np.array(ink, np.float32) * c4[..., None]
    Image.fromarray(np.dstack([res.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)
    return dst


def colour_text_gate(stem, src, dst, poly, measure, ink, gate_thr=0.2, gate_grow=4, wipe_th=0.25, grow=3, a_thr=(0.25, 0.7), sig=(2, 5, 12, 30), seed=37):
    """keep the redraw's own crisp coloured lettering, but only where the original has that colour too (a gate from the original's ink upscaled, grown by gate_grow px):
    the extra outlines, extrusions and glow the redraw drew round the letters are wiped (refilled from the photo round them) and the kept letters are set in one flat `ink`"""
    arr = np.asarray(Image.open(src).convert("RGBA")); H, W = arr.shape[:2]; rgb = arr[..., :3].astype(np.float32)
    o = np.asarray(Image.open(SB / (stem + ".PNG")).convert("RGB")).astype(int)
    pm = poly_mask((H, W), [poly])
    g = cv2.resize(measure(o).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC) > gate_thr
    g = cv2.dilate(g.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gate_grow + 1, 2 * gate_grow + 1))) > 0
    m = measure(arr[..., :3].astype(int)).astype(np.float32)
    a = np.clip((m - a_thr[0]) / (a_thr[1] - a_thr[0]), 0, 1) * g * pm
    a = cv2.GaussianBlur(a, (0, 0), 0.5)
    wm = (m > wipe_th) & pm
    wm = cv2.dilate(wm.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
    fill = _fill(rgb, ~wm, wm, sig) + np.random.default_rng(seed).normal(0, 1.6, (H, W))[..., None]
    w = cv2.GaussianBlur(wm.astype(np.float32), (0, 0), 1.0)[..., None]
    res = rgb * (1 - w) + fill * w
    res = res * (1 - a[..., None]) + np.array(ink, np.float32) * a[..., None]
    Image.fromarray(np.dstack([res.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)
    return dst


def drift_field(stem, rgb, step=40, tw=128, th=112, search=32, min_score=0.85, min_std=12.0, alpha=None):
    """where the redraw's content sits relative to the original's (Lanczos 4x): template matches of blurred brightness on a grid -> samples (x, y, dx, dy, score),
    ref(x + dx, y + dy) ~ redraw(x, y)"""
    k = rgb.shape[1] // Image.open(SB / (stem + ".PNG")).width
    r = cv2.GaussianBlur(cv2.cvtColor(ref_page(stem, k), cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 2.5)
    h = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 2.5)
    H, W = h.shape; S = []
    for y0 in range(search, H - th - search, step):
        for x0 in range(search, W - tw - search, step):
            if alpha is not None and alpha[y0:y0 + th, x0:x0 + tw].min() < 250:
                continue
            T = h[y0:y0 + th, x0:x0 + tw]
            if T.std() < min_std:
                continue
            res = cv2.matchTemplate(r[y0 - search:y0 + th + search, x0 - search:x0 + tw + search], T, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv < min_score or ml[0] in (0, 2 * search) or ml[1] in (0, 2 * search):
                continue
            S.append((x0 + tw / 2, y0 + th / 2, ml[0] - search, ml[1] - search, mv))
    return np.array(S, np.float64)


def smooth_field(samples, shape, sigma=160.0):
    """dense (dx, dy) from the samples: Gaussian-weighted average (normalised convolution of the sample points), median-filtered against outliers first"""
    H, W = shape
    num = np.zeros((2, H, W), np.float32); den = np.zeros((H, W), np.float32)
    xs, ys = samples[:, 0], samples[:, 1]
    keep = np.ones(len(samples), bool)
    for i in range(len(samples)):                                            # drop a sample that disagrees with its neighbours' median by > 4 px
        d = np.hypot(xs - xs[i], ys - ys[i]); nb = (d < 2.2 * sigma) & (d > 0)
        if nb.sum() >= 3:
            md = np.median(samples[nb, 2:4], 0)
            keep[i] = np.abs(samples[i, 2:4] - md).max() <= 4
    for (x, y, dx, dy, s) in samples[keep]:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            num[0, yi, xi] += dx * s; num[1, yi, xi] += dy * s; den[yi, xi] += s
    dn = cv2.GaussianBlur(den, (0, 0), sigma)
    fx = cv2.GaussianBlur(num[0], (0, 0), sigma) / np.maximum(dn, 1e-9)
    fy = cv2.GaussianBlur(num[1], (0, 0), sigma) / np.maximum(dn, 1e-9)
    return fx, fy, keep


def align_to_ref(stem, src, dst, sigma=160.0, **kw):
    """move the redraw's content back onto the original's geometry (the tiled redraw drifted some regions by several px): out(p) = redraw(p - d(p))"""
    arr = np.asarray(Image.open(src).convert("RGBA")); rgb = arr[..., :3]
    smp = drift_field(stem, rgb, alpha=arr[..., 3], **kw)
    fx, fy, keep = smooth_field(smp, rgb.shape[:2], sigma)
    H, W = rgb.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = cv2.remap(rgb, xx - fx, yy - fy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    Image.fromarray(np.dstack([out, arr[..., 3]])).save(dst)
    return smp, keep, fx, fy


def flatten_shading(src, dst, region, flat, protect, k=13, dl=6, db=10, sig=(6, 15, 40, 100), grain=1.6, seed=4):
    """flat comic colour that the redraw shaded (glows, drop shadows, vignettes along the lines): `flat(rgb_int)` picks the clean flat-colour pixels; their normalised-convolution
    fill is the flat colour expected everywhere; broad areas (morphological opening with a k px disc, so thin line art is never caught) that deviate from it by > dl in
    brightness or > db in blue, inside `region` and outside `protect`, are repainted with it"""
    arr = np.asarray(Image.open(src).convert("RGBA")); rgb = arr[..., :3]; f = rgb.astype(int); H, W = rgb.shape[:2]
    ok = flat(f)
    cf = _fill(rgb.astype(np.float32), ok, ~ok, sig)
    Lr = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    Lc = cv2.cvtColor(cf.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    off = ((np.abs(Lr - Lc) > dl) | (np.abs(f[..., 2] - cf[..., 2]) > db)) & ~protect
    broad = cv2.morphologyEx(off.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    todo = broad & region & ~protect
    w = cv2.GaussianBlur(todo.astype(np.float32), (0, 0), 2)[..., None]
    out = rgb * (1 - w) + (cf + np.random.default_rng(seed).normal(0, grain, (H, W))[..., None]) * w
    Image.fromarray(np.dstack([out.clip(0, 255).astype(np.uint8), arr[..., 3]])).save(dst)
    return dst


def lock_flat(stem, src, dst, region, fill_ok, ref_ok, sig=12, chans=(0, 1, 2), amount=1.0):
    """a colour lock for flat comic colour that does not halo round the line art: the low frequencies of the redraw (over its pixels `fill_ok`) and of the original
    (over its pixels `ref_ok`) are taken by normalised convolution, so the dark lines never enter either; their Lab difference is added inside `region` (0..1)"""
    arr = np.asarray(Image.open(src).convert("RGBA")); rgb = arr[..., :3]
    k = rgb.shape[1] // Image.open(SB / (stem + ".PNG")).width
    ref = ref_page(stem, k)
    la = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    lr = cv2.cvtColor(ref.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    def nc(x, m):
        m = m.astype(np.float32)
        return cv2.GaussianBlur(x * m, (0, 0), sig) / np.maximum(cv2.GaussianBlur(m, (0, 0), sig), 1e-4)
    out = la.copy(); reg = region.astype(np.float32)
    for c in chans:
        out[..., c] = la[..., c] + amount * reg * (nc(lr[..., c], ref_ok) - nc(la[..., c], fill_ok))
    res = (cv2.cvtColor(out, cv2.COLOR_Lab2RGB).clip(0, 1) * 255 + .5).astype(np.uint8)
    Image.fromarray(np.dstack([res, arr[..., 3]])).save(dst)
    return dst
