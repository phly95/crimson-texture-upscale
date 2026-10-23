"""mag_parts.py: shared helpers for the part-based magazine covers (pipeline/mag_parts/<COVER>/manifest.json, tools/mag_compose.py).

  plate_up(img, prompt, seed, ...)   a small picture (one figure / photo) -> 4x by two klein-edit stages (2x Lanczos -> klein -> colour lock), one tile each,
                                     so the model sees only that picture, not the page.  Needs the klein server.
  consistency(hd, orig)              mean abs error + worst blocks of the HD part box-downscaled to the original's size against the original
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tile_upscale as tu                                                        # noqa: E402

PARTS = ROOT / "pipeline" / "mag_parts"
SCRAPBOOK = Path.home() / "Games/CSVM/extracted/rof/ASSETS/GRAPHICS/SCRAPBOOK"


def f32(im):
    return np.asarray(im.convert("RGB")).astype(np.float32) / 255


def u8(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255 + .5).astype(np.uint8))


def _np(x, w, sg):
    """normalised blur: blur(x*w)/blur(w) (the average of x over the weight w only)"""
    num = cv2.GaussianBlur(x * w[..., None], (0, 0), sg).reshape(x.shape)
    den = cv2.GaussianBlur(w, (0, 0), sg)[..., None]
    return num / np.maximum(den, 1e-4)


def region_lock(gen, ref, m, sigma_l, sigma_ab, k_l=0.9):
    """tu.colour_lock that does not blur across the matte edge: figure (m) and backdrop (1-m) are pulled back to ref separately, so a sharp redraw edge against a soft
    source edge does not leave a lighter/tinted halo.  gen/ref float RGB, m float 0..1"""
    from skimage import color as skc
    lg, lr = skc.rgb2lab(gen).astype(np.float32), skc.rgb2lab(ref).astype(np.float32)
    d = np.zeros_like(lg)
    for w in (m.astype(np.float32), 1 - m.astype(np.float32)):
        w = cv2.erode((w > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)   # keep the transition pixels out of both averages
        dl = _np(lr[..., :1], w, sigma_l) - _np(lg[..., :1], w, sigma_l)
        dab = _np(lr[..., 1:], w, sigma_ab) - _np(lg[..., 1:], w, sigma_ab)
        d += w[..., None] * np.concatenate([k_l * dl, dab], -1)
    # transition pixels (in neither mask) take the mean of their neighbours' correction
    wsum = cv2.erode((m > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32) + cv2.erode((m <= 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    hole = (wsum < 0.5).astype(np.float32)
    if hole.any():
        fill = cv2.GaussianBlur(d * (1 - hole)[..., None], (0, 0), 3) / np.maximum(cv2.GaussianBlur(1 - hole, (0, 0), 3), 1e-4)[..., None]
        d = np.where(hole[..., None] > 0, fill, d)
    return skc.lab2rgb((lg + d).astype(np.float64)).clip(0, 1).astype(np.float32)


def one_stage(img, prompt, seed, tag, sigma_l=8.0, sigma_ab=24.0, tmp=Path("/tmp"), matte=None):
    """2x Lanczos then one klein edit of the whole picture (W,H multiples of 16) with the broad tone locked to the Lanczos version -> (locked, raw, lanczos)"""
    W, H = img.size[0] * 2, img.size[1] * 2
    assert W % 16 == 0 and H % 16 == 0, (W, H)
    base = img.convert("RGB").resize((W, H), Image.LANCZOS)
    ref_p, out_p = tmp / f"_plate_{tag}.png", tmp / f"_plate_{tag}_out.png"
    base.save(ref_p)
    tu.locked_gen_wh(ref_p, out_p, prompt, W, H, seed)
    raw = Image.open(out_p).convert("RGB").resize((W, H), Image.LANCZOS)
    tu.HF_CLIP = False
    if matte:                                                                     # matte = (rect, core) in this stage's source px: edge-aware lock
        k = W / img.size[0]
        m = figure_matte(raw, tuple(v * k for v in matte[0]), fg_core=tuple(v * k for v in matte[1]) if matte[1] else None)
        out = region_lock(f32(raw), f32(base), m, sigma_l, sigma_ab)
    else:
        out = tu.colour_lock(f32(raw), f32(base), sigma_l, sigma_ab)
    return u8(out), raw, base


def plate_up(img, prompt, seed=100, tag="plate", sigma_l=8.0, sigma_ab=24.0, keep=None, matte=None):
    """4x of a small picture; returns the final PIL image (and writes the intermediate stages to keep/ if given)"""
    sl = sigma_l if isinstance(sigma_l, (tuple, list)) else (sigma_l, sigma_l)       # per-stage lock sigmas
    sa = sigma_ab if isinstance(sigma_ab, (tuple, list)) else (sigma_ab, sigma_ab)
    s1, raw1, _ = one_stage(img, prompt, seed, tag + "1", sl[0], sa[0], matte=matte)
    m2 = None if not matte else tuple(None if v is None else tuple(c * 2 for c in v) for v in matte)
    s2, raw2, _ = one_stage(s1, prompt, seed + 1, tag + "2", sl[1], sa[1], matte=m2)
    if keep:
        Path(keep).mkdir(parents=True, exist_ok=True)
        for n, im in (("s1", s1), ("raw1", raw1), ("s2", s2), ("raw2", raw2)):
            im.save(Path(keep) / f"{tag}_{n}.png")
    return s2


def consistency(hd, orig, blocks=8):
    """HD part box-downscaled to the original's size vs the original: (mean abs error 0..255, [(err, x, y)] of the worst blocks in orig px)"""
    a = np.asarray(hd.convert("RGB").resize(orig.size, Image.BOX)).astype(np.float32)
    b = np.asarray(orig.convert("RGB")).astype(np.float32)
    d = np.abs(a - b).mean(-1)
    h, w = d.shape
    cells = sorted(((float(d[y:y + blocks, x:x + blocks].mean()), x, y) for y in range(0, h - blocks + 1, blocks) for x in range(0, w - blocks + 1, blocks)), reverse=True)
    return float(d.mean()), cells[:5]


def gen_refs(refs, out, prompt, W, H, seed, timeout=300):
    """klein edit with SEVERAL reference images (first = the picture to edit, the others = identity / look references); serialised like tu.locked_gen_wh"""
    import base64, fcntl, time
    ks = tu.ks
    with open("/tmp/klein_tile.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not ks.up():
            ks.start(ROOT / "pipeline" / "klein_server.log")
        b64 = [base64.b64encode(Path(r).read_bytes()).decode() for r in refs]
        r = ks._call("/sdcpp/v1/img_gen", {"prompt": prompt, "width": int(W), "height": int(H), "seed": int(seed), "ref_images": b64, "output_format": "png"}, timeout=30)
        return ks.collect(r["id"], out, timeout)


def strokes_mask(rgb, poly=None, k=7, th=30, bright=True, grow=1):
    """thin light (or dark) strokes (lettering) as a 0/1 mask: top-hat of the luminance with a k px kernel; limited to POLY (list of (x,y)) if given"""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    th_img = cv2.morphologyEx(g, cv2.MORPH_TOPHAT if bright else cv2.MORPH_BLACKHAT, ker)
    m = (th_img > th).astype(np.uint8)
    if poly is not None:
        pm = np.zeros(m.shape, np.uint8); cv2.fillPoly(pm, [np.array(poly, np.int32)], 1); m &= pm
    return cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) if grow else m


def inpaint(rgb, mask, r=4, ns=False):
    return cv2.inpaint(rgb, mask.astype(np.uint8), r, cv2.INPAINT_NS if ns else cv2.INPAINT_TELEA)


def opaque_on_grey(path, grey=110):
    o = Image.open(path).convert("RGBA")
    return np.asarray(Image.alpha_composite(Image.new("RGBA", o.size, (grey,) * 3 + (255,)), o).convert("RGB"))


def figure_matte(img, rect, iters=6, fg_core=None, scale=0.5):
    """GrabCut matte of a figure on its backdrop: IMG PIL RGB, RECT (x0,y0,x1,y1) loosely around the figure -> float32 0..1 matte (full size), run at SCALE for speed.
    FG_CORE (x0,y0,x1,y1) marks a box that is surely figure."""
    a = np.asarray(img.convert("RGB"))
    s = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    m = np.full(s.shape[:2], cv2.GC_BGD, np.uint8)
    x0, y0, x1, y1 = [int(v * scale) for v in rect]
    m[y0:y1, x0:x1] = cv2.GC_PR_FGD
    if fg_core:
        c = [int(v * scale) for v in fg_core]; m[c[1]:c[3], c[0]:c[2]] = cv2.GC_FGD
    bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(s, m, None, bg, fg, iters, cv2.GC_INIT_WITH_MASK)
    f = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    f = cv2.morphologyEx(f, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(f)
    if n > 1:
        f = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    f = cv2.resize(f.astype(np.float32), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    return f


# ---- manifests ------------------------------------------------------------------------------------------------------------------------------------------------
import json, shutil

UIDS = {"MAG1": "scrap_SB_00_00_MAG1__37e18d0c", "MAG2": "scrap_SB_01_02_MAG2__12bf7da1", "MAG3": "scrap_SB_11_01_MAG3__174eaf47", "MAG4": "scrap_SB_11_01_MAG4__39635161",
        "MAG5": "scrap_SB_14_01_MAG5__b1523eb3", "MAG7": "scrap_SB_21_01_MAG7__6c9a2cef", "ZPJCOVER": "scrap_SB_19_01_ZPJCOVER__9b658f3c", "COMICOVER": "scrap_SB_22_01_COMICOVER__2c0858c3"}
STEMS = {"MAG1": "SB_00_00_MAG1", "MAG2": "SB_01_02_MAG2", "MAG3": "SB_11_01_MAG3", "MAG4": "SB_11_01_MAG4", "MAG5": "SB_14_01_MAG5", "MAG7": "SB_21_01_MAG7",
         "ZPJCOVER": "SB_19_01_ZPJCOVER", "COMICOVER": "SB_22_01_COMICOVER"}


class Manifest:
    """pipeline/mag_parts/<COVER>/manifest.json: {cover, uid, size, alpha, orig, parts:[{id,name,kind,z,file,x,y,w,h,source:{type,method,tool,orig_box,shared},status,qa:{result,note}}]}
    kind: background | figure | photo | emblem | text | rule | decor.  Part files are RGBA PNGs at 4x page scale, bbox-sized; x,y = top-left on the page."""

    def __init__(self, cover):
        self.cover, self.dir = cover, PARTS / cover
        self.path = self.dir / "manifest.json"
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {"cover": cover, "uid": UIDS[cover], "size": None, "alpha": "alpha.png", "orig": "orig.png", "parts": []}

    def save(self):
        self.d["parts"].sort(key=lambda p: (p["z"], p["id"]))
        self.path.write_text(json.dumps(self.d, indent=1))

    def init(self, final_png, orig_png):
        """start a cover: page alpha, original and the current approved final as the 'page' background part"""
        (self.dir / "parts").mkdir(parents=True, exist_ok=True)
        fin = Image.open(final_png).convert("RGBA")
        self.d["size"] = list(fin.size)
        fin.getchannel("A").save(self.dir / "alpha.png")
        shutil.copy(orig_png, self.dir / "orig.png")
        bg = Image.new("RGBA", fin.size, (0, 0, 0, 255)); bg.alpha_composite(fin)
        self.add("page", "Page (approved redraw)", "background", bg, 0, 0, 0, {"type": "upscaled", "method": "deskew, tiled FLUX Klein 4x, cleanup (earlier rounds)", "tool": "tools/tile_mags_rot.py + tools/mag_finish.py"}, "fine", "pass", "base layer; problem elements are laid over it as separate parts")

    def add(self, pid, name, kind, rgba, x, y, z, source, status="fine", qa="pending", note=""):
        (self.dir / "parts").mkdir(parents=True, exist_ok=True)
        f = f"parts/{pid}.png"
        rgba.save(self.dir / f)
        ent = {"id": pid, "name": name, "kind": kind, "z": z, "file": f, "x": int(x), "y": int(y), "w": rgba.width, "h": rgba.height, "source": source, "status": status, "qa": {"result": qa, "note": note}}
        self.d["parts"] = [p for p in self.d["parts"] if p["id"] != pid] + [ent]
        self.save()
        return ent

    def part(self, pid):
        return next(p for p in self.d["parts"] if p["id"] == pid)

    def set_qa(self, pid, result, note=""):
        self.part(pid)["qa"] = {"result": result, "note": note}; self.save()


def compose(cover, out=None, skip=()):
    """deterministic stack of the manifest's parts (z order) -> RGBA page with the original page alpha; writes OUT if given"""
    mf = Manifest(cover); W, H = mf.d["size"]
    cv = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    for p in sorted(mf.d["parts"], key=lambda p: (p["z"], p["id"])):
        if p["id"] in skip:
            continue
        for poly in p.get("wipe_gold", []):                    # gold (also dim ghost copies of gold) letters of what is below, inside the polygon, painted out first
            cv = wipe_colour(cv, poly)
        for poly in p.get("clears", []):                       # polygons (page px) of what is under this part that must not show: filled from the surroundings first
            cv = clear_poly(cv, poly)
        cv.alpha_composite(Image.open(mf.dir / p["file"]).convert("RGBA"), (p["x"], p["y"]))
    cv.putalpha(Image.open(mf.dir / mf.d["alpha"]).convert("L"))
    if out:
        cv.save(out)
    return cv


def figure_rgba(plate, rect, core, erode=3, blur=1.8, fade=None, matte=None, cut=(), grow_xy=None):
    """cut a finished plate (PIL RGB, 4x) out with its GrabCut matte, slightly eroded + softened, -> RGBA.  FADE = {'b': px, 't': px, 'l': px, 'r': px}: sides where the figure is cut by
    the plate border (it continues underneath) fade out over that many px instead of ending at the border.  MATTE = a precomputed float matte"""
    m = matte if matte is not None else figure_matte(plate, rect, fg_core=core)
    m = (m > 0.5).astype(np.uint8)
    for poly in cut:                                                                  # polygons (plate px) that are NOT figure although the matte took them (backdrop pockets)
        cv2.fillPoly(m, [np.array(poly, np.int32)], 0)
    if grow_xy:                                                                        # anisotropic grow (gx, gy) px: an elliptical band of the plate's backdrop round the figure
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_xy[0] + 1, 2 * grow_xy[1] + 1)))
    elif erode > 0:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1)))
    elif erode < 0:                                                                    # negative = grow: keep a band of the plate's own backdrop round the figure
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (-2 * erode + 1, -2 * erode + 1)))
    a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), blur)
    h, w = a.shape
    for side, n in (fade or {}).items():
        ramp = np.linspace(0, 1, n, dtype=np.float32) ** 1.5
        if side == "b": a[h - n:] *= ramp[::-1, None]
        if side == "t": a[:n] *= ramp[:, None]
        if side == "l": a[:, :n] *= ramp[None, :]
        if side == "r": a[:, w - n:] *= ramp[None, ::-1]
    return Image.merge("RGBA", (*plate.convert("RGB").split(), Image.fromarray((a * 255 + .5).astype(np.uint8))))


WIN = Path.home() / ".local/share/fonts/winfonts"


def text_layer(size, text, x0, base0, x1, slope, cap, font, ink, shadow=0.0, stroke=0, space=2.4, soft=0.5, weight=1.0, shadow_off=(2, 3), shadow_blur=1.6, margin=10):
    """lettering as its own transparent layer: TEXT set in FONT along the line through (x0, base0) with slope dy/dx, spanning x0..x1 with capitals CAP px high (letter-spaced to fit), colour INK,
    optional soft dark SHADOW.  SIZE = page (W, H).  Returns (RGBA bbox image, x, y)  (same geometry as mag_polish.tilted_text)"""
    from PIL import ImageDraw, ImageFont
    W, H = size
    ang = np.degrees(np.arctan(slope)); L = (x1 - x0) / np.cos(np.arctan(slope))
    f0 = ImageFont.truetype(str(font), 200); hb = f0.getbbox("H"); fs = 200 * cap / (hb[3] - hb[1]); f = ImageFont.truetype(str(font), int(round(fs)))
    widths = [f.getlength(c) * (space if c == ' ' else 1.0) for c in text]; gap = (L - sum(widths)) / max(len(text) - 1, 1)
    lay = Image.new("L", (int(L) + 40, int(cap * 3)), 0); d = ImageDraw.Draw(lay); x = 20.0
    for c, w in zip(text, widths):
        d.text((x, cap * 2), c, font=f, fill=255, anchor="ls", stroke_width=stroke, stroke_fill=255); x += w + gap
    if weight != 1.0:
        lay = Image.fromarray(cv2.GaussianBlur(np.asarray(lay).astype(np.float32), (0, 0), 0.6).clip(0, 255).astype(np.uint8))
    M = cv2.getRotationMatrix2D((20, cap * 2), -ang, 1.0)
    M[0, 2] += x0 - 20; M[1, 2] += base0 - cap * 2
    a = cv2.warpAffine(np.asarray(lay), M, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
    al = cv2.GaussianBlur(a, (0, 0), soft)
    sh = cv2.GaussianBlur(np.roll(np.roll(a, shadow_off[1], 0), shadow_off[0], 1), (0, 0), shadow_blur) * shadow if shadow else np.zeros_like(a)
    ys, xs = np.nonzero((al > 0.003) | (sh > 0.003))
    bx0, by0, bx1, by1 = max(xs.min() - margin, 0), max(ys.min() - margin, 0), min(xs.max() + margin + 1, W), min(ys.max() + margin + 1, H)
    al, sh = al[by0:by1, bx0:bx1], sh[by0:by1, bx0:bx1]
    # straight-alpha composite of: shadow (dark) under ink
    A = 1 - (1 - al) * (1 - sh)
    ink_a = np.array(ink, np.float32); dark = np.array((15, 10, 8), np.float32)
    rgb = (ink_a * al[..., None] + dark * (sh * (1 - al))[..., None]) / np.maximum(A, 1e-6)[..., None]
    return Image.fromarray(np.dstack([rgb.clip(0, 255), A * 255]).astype(np.uint8), "RGBA"), int(bx0), int(by0)


def sepia_project(plate, ref, k=0.9, match=True, mask=None):
    """a toned (sepia) photograph has one hue: project the plate's Lab a/b onto the hue axis of the source REF (same size, PIL RGB), k = how far (1 = strictly monochrome tone),
    and, if MATCH, scale the plate's chroma so its 90th percentile equals the source's.  Removes the blue/teal/pink tints a redraw invents in the shadows."""
    from skimage import color as skc
    lp, lr = skc.rgb2lab(f32(plate)).astype(np.float32), skc.rgb2lab(f32(ref)).astype(np.float32)
    sel_px = np.ones(lr.shape[:2], bool) if mask is None else mask.astype(bool)
    ab = lr[..., 1:][sel_px].reshape(-1, 2); ch = np.hypot(ab[:, 0], ab[:, 1]); sel = ab[ch > 2]
    u = sel.mean(0) if len(sel) else np.array([1.0, 1.0]); u = u / np.hypot(*u)
    pr = lr[..., 1] * u[0] + lr[..., 2] * u[1]
    pp = lp[..., 1] * u[0] + lp[..., 2] * u[1]
    s = 1.0
    if match:
        s = float(np.clip(np.percentile(pr[sel_px], 90) / max(np.percentile(pp[sel_px], 90), 1e-3), 0.6, 2.2))
    new = np.stack([np.maximum(pp, 0) * s * u[0], np.maximum(pp, 0) * s * u[1]], -1)                 # negative (blue/purple) projections become neutral, not their opposite
    lp[..., 1:] = (1 - k) * lp[..., 1:] + k * new
    return u8(skc.lab2rgb(lp.astype(np.float64)).clip(0, 1))


def orig_rgb(cover):
    """the cover's original as RGB on the page's own colours (alpha ignored) and its alpha"""
    o = Image.open(SCRAPBOOK / (STEMS[cover] + ".PNG")).convert("RGBA")
    return o.convert("RGB"), o.getchannel("A")


def crisp(img, k=4, amount=1.3, radius=1.6, steep=0.0):
    """the honest upscale of a small original detail: Lanczos k x, then unsharp mask (RADIUS px at the new scale, AMOUNT) and, if STEEP>0, a shock-filter-like edge steepening.
    Invents nothing: every stroke is the original's."""
    big = np.asarray(img.convert("RGB").resize((img.width * k, img.height * k), Image.LANCZOS)).astype(np.float32)
    bl = cv2.GaussianBlur(big, (0, 0), radius)
    out = big + amount * (big - bl)
    if steep:
        g = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        lap = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.2), cv2.CV_32F)
        out = out - steep * np.sign(lap)[..., None] * 6 * (np.abs(lap)[..., None] > 1)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def box_part(cover, box, feather=6, k=4, amount=1.3, radius=1.6, steep=0.0, poly=None, grain=0.0, seed=1, rgb=None):
    """a patch of the ORIGINAL (BOX in original px) upscaled with crisp() -> (RGBA, x, y) at page scale.  alpha: the box (or POLY in orig px) eroded by FEATHER px and blurred"""
    if rgb is None:
        rgb, _ = orig_rgb(cover)
    x0, y0, x1, y1 = box
    # take a margin for the filters, then crop
    mg = 4
    c = rgb.crop((x0 - mg, y0 - mg, x1 + mg, y1 + mg))
    big = crisp(c, k, amount, radius, steep).crop((mg * k, mg * k, mg * k + (x1 - x0) * k, mg * k + (y1 - y0) * k))
    w, h = big.size
    m = np.zeros((h, w), np.float32)
    if poly is None:
        m[:] = 1
        m = cv2.copyMakeBorder(m[feather:-feather, feather:-feather], feather, feather, feather, feather, cv2.BORDER_CONSTANT, value=0) if feather else m
    else:
        cv2.fillPoly(m, [np.array([((px - x0) * k, (py - y0) * k) for px, py in poly], np.int32)], 1.0)
    a = cv2.GaussianBlur(m, (0, 0), max(feather / 2, 0.5))
    arr = np.asarray(big).astype(np.float32)
    if grain:
        arr = arr + np.random.default_rng(seed).normal(0, grain, arr.shape[:2])[..., None]
    return Image.fromarray(np.dstack([arr.clip(0, 255), a * 255]).astype(np.uint8), "RGBA"), x0 * k, y0 * k


def trace_up(img, k=4, blur=0.8, lo=None, hi=None, steep=7.0, mid=None, dark=None, light=None, up_blur=1.2):
    """a two-tone graphic (logo, badge) from small original pixels, traced rather than painted: Lanczos k x, luminance through a steep sigmoid around MID (default: mean of the 10th/90th
    percentile) so edges become clean contours; the output colours come from the original's own dark/light ends (percentiles) unless DARK/LIGHT (RGB) are given.  Invents no shapes."""
    big = img.convert("RGB").resize((img.width * k, img.height * k), Image.LANCZOS)
    a = np.asarray(big).astype(np.float32)
    g = cv2.GaussianBlur(cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), up_blur * k / 4)
    p10, p90 = np.percentile(g, 10), np.percentile(g, 90)
    mid = (p10 + p90) / 2 if mid is None else mid
    t = 1 / (1 + np.exp(-(g - mid) / max((p90 - p10) / (2 * steep), 1e-3)))
    t = cv2.GaussianBlur(t, (0, 0), blur)
    msk_d, msk_l = g <= p10 + 4, g >= p90 - 4
    d = np.array(dark if dark is not None else a[msk_d].mean(0), np.float32); l = np.array(light if light is not None else a[msk_l].mean(0), np.float32)
    return Image.fromarray((d[None, None] * (1 - t[..., None]) + l[None, None] * t[..., None]).clip(0, 255).astype(np.uint8))


def traced_part(cover, box, poly, k=4, steep=5.0, dark=None, light=None, min_contrast=22, grow=1.5, feather=1.2, up_blur=1.2, mid=None):
    """a logo / badge of a cover as its own part: BOX (orig px) traced with trace_up(); the part's alpha = POLY (orig px) intersected with where the original differs from its local backdrop
    (median of the box border) by MIN_CONTRAST, closed, holes filled, grown GROW px (page scale), softened -> (RGBA, x, y)"""
    rgb, _ = orig_rgb(cover)
    x0, y0, x1, y1 = box
    mg = 3
    crop = rgb.crop((x0 - mg, y0 - mg, x1 + mg, y1 + mg))
    tr = trace_up(crop, k, steep=steep, dark=dark, light=light, up_blur=up_blur, mid=mid).crop((mg * k, mg * k, mg * k + (x1 - x0) * k, mg * k + (y1 - y0) * k))
    g = cv2.cvtColor(np.asarray(rgb.crop(box)), cv2.COLOR_RGB2GRAY).astype(np.float32)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    diff = (np.abs(g - np.median(border)) > min_contrast).astype(np.uint8)
    pm = np.zeros(g.shape, np.uint8); cv2.fillPoly(pm, [np.array([(px - x0, py - y0) for px, py in poly], np.int32)], 1)
    m = cv2.resize(diff & pm, (g.shape[1] * k, g.shape[0] * k), interpolation=cv2.INTER_NEAREST)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    if cnts:
        cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
    filled = cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(grow) + 1, 2 * int(grow) + 1))) if grow else filled
    a = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), feather)
    return Image.merge("RGBA", (*tr.split(), Image.fromarray((a * 255 + .5).astype(np.uint8)))), x0 * k, y0 * k


def clear_poly(cv, poly, r=6, grow=3, feather=2.0):
    """fill the polygon (page px) of RGBA canvas CV from its surroundings (TELEA inpaint at half scale, soft edge) so an older element underneath cannot peek out round a new part"""
    a = np.asarray(cv.convert("RGB")).copy()
    H, W = a.shape[:2]
    m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    m = cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    ys, xs = np.nonzero(m); pad = 24
    x0, x1, y0, y1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, W), max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)
    sub, sm = a[y0:y1, x0:x1], m[y0:y1, x0:x1]
    h, w = sub.shape[:2]
    half = cv2.resize(sub, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    hm = cv2.resize(sm, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    fill = cv2.resize(cv2.inpaint(half, hm, r, cv2.INPAINT_TELEA), (w, h), interpolation=cv2.INTER_CUBIC)
    sf = cv2.GaussianBlur(sm.astype(np.float32), (0, 0), feather)[..., None]
    sub = (sub * (1 - sf) + fill * sf).clip(0, 255).astype(np.uint8)
    a[y0:y1, x0:x1] = sub
    out = Image.fromarray(a).convert("RGBA"); out.putalpha(cv.getchannel("A"))
    return out


def tone_match(plate, ref, matte=None, strength=0.5, bins=256):
    """partial lightness histogram match of PLATE to REF (same size, PIL RGB), over MATTE (float, optional): quantile mapping of L blended STRENGTH of the way, so a redraw's modern
    contrast (deep blacks, bright highlights) is pulled toward the printed original's tonal range without changing its structure"""
    from skimage import color as skc
    lp, lr = skc.rgb2lab(f32(plate)).astype(np.float32), skc.rgb2lab(f32(ref)).astype(np.float32)
    sel = np.ones(lp.shape[:2], bool) if matte is None else matte > 0.5
    qp = np.percentile(lp[..., 0][sel], np.linspace(0, 100, bins)); qr = np.percentile(lr[..., 0][sel], np.linspace(0, 100, bins))
    mapped = np.interp(lp[..., 0], qp, qr)
    lp[..., 0] = (1 - strength) * lp[..., 0] + strength * mapped
    return u8(skc.lab2rgb(lp.astype(np.float64)).clip(0, 1))


def layer_part(alpha, ink, margin=8, grain=0.0):
    """a float 0..1 page-sized alpha layer + flat ink colour -> (RGBA bbox crop, x, y)"""
    ys, xs = np.nonzero(alpha > 0.003)
    x0, y0, x1, y1 = max(xs.min() - margin, 0), max(ys.min() - margin, 0), min(xs.max() + margin + 1, alpha.shape[1]), min(ys.max() + margin + 1, alpha.shape[0])
    a = alpha[y0:y1, x0:x1]
    rgb = np.empty(a.shape + (3,), np.uint8); rgb[:] = ink
    return Image.fromarray(np.dstack([rgb, (np.clip(a, 0, 1) * 255 + .5).astype(np.uint8)]), "RGBA"), int(x0), int(y0)


def backdrop_patch(cover, placed, grow=14, feather=5.0, k=4, rgb=None, inpaint_r=4):
    """where small icons are placed over the old ones: the original's own backdrop with the old icons inpainted out (at ORIGINAL resolution, where the backdrop is smooth), Lanczos k x, as a
    part whose alpha is the placed icons' footprint grown by GROW page px and softened.  PLACED = [(RGBA, x, y)] at page scale -> (RGBA bbox, x, y)"""
    base_rgb = np.asarray(orig_rgb(cover)[0] if rgb is None else rgb)
    H, W = base_rgb.shape[:2]
    m4 = np.zeros((H * k, W * k), np.uint8)
    for im, x, y in placed:
        a = (np.asarray(im.getchannel("A")) > 10).astype(np.uint8)
        m4[y:y + a.shape[0], x:x + a.shape[1]] |= a[:m4.shape[0] - y, :m4.shape[1] - x]
    grown = cv2.dilate(m4, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    m1 = (cv2.resize(grown.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) > 0.05).astype(np.uint8)
    m1 = cv2.dilate(m1, np.ones((3, 3), np.uint8))
    clean = cv2.inpaint(base_rgb.copy(), m1, inpaint_r, cv2.INPAINT_NS)
    big = np.asarray(Image.fromarray(clean).resize((W * k, H * k), Image.LANCZOS))
    al = cv2.GaussianBlur(grown.astype(np.float32), (0, 0), feather)
    ys, xs = np.nonzero(al > 0.01)
    x0, y0, x1, y1 = max(xs.min() - 4, 0), max(ys.min() - 4, 0), min(xs.max() + 5, W * k), min(ys.max() + 5, H * k)
    return Image.fromarray(np.dstack([big[y0:y1, x0:x1], (np.clip(al[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8)]), "RGBA"), int(x0), int(y0)


def photo_mask(rgb_img, purple_gap=28):
    """a small tipped-on photo on coloured paper: the non-paper pixels (paper = clearly purple: blue and red both above green by PURPLE_GAP) -> (largest component's convex hull as uint8 mask)"""
    f = np.asarray(rgb_img.convert("RGB")).astype(np.int16)
    paper = (f[..., 2] > f[..., 1] + purple_gap) & (f[..., 0] > f[..., 1] + purple_gap)
    ph = (~paper).astype(np.uint8)
    ph = cv2.morphologyEx(ph, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ph)
    if n < 2:
        return ph
    comp = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    out = np.zeros_like(comp); cv2.fillConvexPoly(out, hull, 1)
    return out


def wipe_colour(cv, poly, gb=40, rb=55, grow=3, r=4):
    """inpaint away the gold-ish pixels (g-b > GB and r-b > RB: lettering and its dim copies; sepia photo tones are g-b ~ 20) of RGBA canvas CV inside POLY (page px)"""
    a = np.asarray(cv.convert("RGB")).copy(); f = a.astype(np.int16)
    m = ((f[..., 1] - f[..., 2] > gb) & (f[..., 0] - f[..., 2] > rb)).astype(np.uint8)
    pm = np.zeros(m.shape, np.uint8); cv2.fillPoly(pm, [np.array(poly, np.int32)], 1)
    m = cv2.dilate(m & pm, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) & pm
    if m.any():
        a = cv2.inpaint(a, m, r, cv2.INPAINT_TELEA)
    out = Image.fromarray(a).convert("RGBA"); out.putalpha(cv.getchannel("A"))
    return out


# ---- covers drawn in the deskewed ("upright") frame of tile_mags_rot ----------------------------------------------------------------------------------------------
import tile_mags_rot as tr


def upright_1x(cover):
    """the original deskewed (tile_mags_rot frame, 1x, edge-extended beyond the paper) as RGB PIL image"""
    stem = STEMS[cover]
    im = Image.open(SCRAPBOOK / (stem + ".PNG")); W, H = im.size
    M, (W2, H2) = tr.forward(W, H, tr.ANGLE[stem])
    rgb = cv2.warpAffine(tr.extend(np.asarray(im)), M, (W2, H2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rgb)


def to_page(rgba_up, cover, x_up=0, y_up=0, margin=4):
    """an RGBA picture laid out in the upright frame at 4x (its top-left at (x_up, y_up) in UPRIGHT 4X px) -> (RGBA bbox crop in the page frame at 4x, x, y)"""
    stem = STEMS[cover]
    im = Image.open(SCRAPBOOK / (stem + ".PNG")); W, H = im.size
    M, (W2, H2) = tr.forward(W, H, tr.ANGLE[stem])
    Sm = np.diag([4.0, 4.0, 1.0])
    M4 = (Sm @ np.vstack([M, [0, 0, 1]]) @ np.linalg.inv(Sm))[:2]
    full = np.zeros((H2 * 4, W2 * 4, 4), np.float32)
    a = np.asarray(rgba_up.convert("RGBA")).astype(np.float32)
    h, w = a.shape[:2]
    full[y_up:y_up + h, x_up:x_up + w] = a
    pre = full.copy(); pre[..., :3] *= (full[..., 3:4] / 255.0)                        # premultiplied: no dark fringe from the transparent surround
    wp = cv2.warpAffine(pre, M4, (W * 4, H * 4), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    alpha = wp[..., 3]
    rgb = np.where(alpha[..., None] > 0.5, wp[..., :3] / np.maximum(alpha[..., None] / 255.0, 1e-3), 0)
    ys, xs = np.nonzero(alpha > 1)
    x0, y0, x1, y1 = max(xs.min() - margin, 0), max(ys.min() - margin, 0), min(xs.max() + margin + 1, W * 4), min(ys.max() + margin + 1, H * 4)
    out = np.dstack([rgb, alpha]).clip(0, 255).astype(np.uint8)[y0:y1, x0:x1]
    return Image.fromarray(out, "RGBA"), int(x0), int(y0)


def detail_transfer(plate, ref, sigma=3.0, k=0.9, edge_gate=None):
    """keep the ORIGINAL's structure and take only the redraw's fine detail: ref (Lanczos) + k * (plate - blur(plate, sigma)), per channel at the plate's scale.  Shapes larger than ~sigma px are the
    original's; the redraw can sharpen edges and add grain/texture but cannot add a landing gear.  EDGE_GATE (0..1 float): scale the detail down where the original is smooth"""
    p, r = f32(plate), f32(ref)
    hf = p - cv2.GaussianBlur(p, (0, 0), sigma)
    if edge_gate is not None:
        hf = hf * edge_gate[..., None]
    return u8(r + k * hf)


def flat_up(img, k=4, sigma_s=3.0, sigma_r=0.08, amount=0.9, radius=1.5, steep=0.5):
    """the honest upscale of flat-colour comic art: Lanczos k x, an edge-preserving smooth (colour bands lose their JPEG/blur ripple, lines stay), then unsharp + edge steepening so the ink lines are
    crisp.  Nothing is invented."""
    big = np.asarray(img.convert("RGB").resize((img.width * k, img.height * k), Image.LANCZOS))
    sm = cv2.bilateralFilter(big, 0, sigma_r * 255, sigma_s * k / 2)
    return crisp(Image.fromarray(sm), 1, amount, radius, steep)


def orig_ext_rgb(cover):
    """the original's RGB with the colour under its transparent part replaced by a smooth extension of the paper edge (tile_mags_rot.extend): no grey/dark bleed at torn edges"""
    im = Image.open(SCRAPBOOK / (STEMS[cover] + ".PNG")).convert("RGBA")
    return Image.fromarray(tr.extend(np.asarray(im)))


def alpha_clip(rgba, cover, x, y, erode=1):
    """multiply RGBA part's alpha (placed at x, y, 4x) by the original page alpha (Lanczos 4x, eroded ERODE px) so a patch cannot paint outside the paper"""
    a = Image.open(SCRAPBOOK / (STEMS[cover] + ".PNG")).convert("RGBA").getchannel("A")
    a4 = np.asarray(a.resize((a.width * 4, a.height * 4), Image.LANCZOS)).astype(np.float32) / 255
    if erode:
        a4 = cv2.erode(a4, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    a4 = cv2.GaussianBlur(a4, (0, 0), 1.0)
    w, h = rgba.size
    arr = np.asarray(rgba).astype(np.float32)
    sub = a4[y:y + h, x:x + w]
    arr[:sub.shape[0], :sub.shape[1], 3] *= sub
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA")


def fold_mask_1x(cover, min_ch=165, min_area=250):
    """big bright connected areas of the original (the white page curl / fold): 0/1 uint8 at original size"""
    rgb = np.asarray(orig_rgb(cover)[0])
    m = (rgb.min(-1) > min_ch).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out


def clip_to_fold(rgba, cover, x, y, grow=1):
    """zero the part's alpha wherever the original shows the fold (so an icon that sits under a page curl stays under it)"""
    f = fold_mask_1x(cover)
    f4 = cv2.resize(f.astype(np.float32), (f.shape[1] * 4, f.shape[0] * 4), interpolation=cv2.INTER_LINEAR)
    f4 = cv2.GaussianBlur((f4 > 0.5).astype(np.float32), (0, 0), 1.2)
    w, h = rgba.size
    arr = np.asarray(rgba).astype(np.float32)
    sub = f4[y:y + h, x:x + w]
    arr[:sub.shape[0], :sub.shape[1], 3] *= (1 - sub)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA")
