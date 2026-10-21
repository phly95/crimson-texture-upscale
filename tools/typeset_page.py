"""typeset_page.py: re-set the typed text of a scrapbook thumbnail from the game's own string.
The thumbnails carry their text baked in at ~8 px, so the redraw can only invent it (garbled words). The game shows the real string when the item is clicked
(ui_strings.json, COUR14 = Courier New ...). Method: straighten the page (deskew by the printed lines), find the printed lines of the region that is text, wrap the real
paragraphs to the same number of lines, wipe the old ink (paper estimated from the unprinted paper round it) and lay the new lines on in the game's own font, then undo the
straightening for the text layer only, so the rest of the picture is not resampled.
  typeset_page.py analyze NAME IN.png [y0 y1]        print the skew and the printed lines found (fractions of the height)
  typeset_page.py run NAME IN.png OUT.png            NAME is a key of ITEMS
"""
import json, re, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

STR = {d["id"]: d["text"] for d in json.load(open(Path.home() / "Games/CSVM/extracted/rof/ui_strings.json"))}
F = Path.home() / ".local/share/fonts/winfonts"
FONTS = {"COUR": (F / "cour.ttf", F / "courbd.ttf", 0.6), "FREE": (F / "FREESCPT.TTF", None, None), "PEPITA": (F / "PEPITA.TTF", None, None), "VIN": (F / "VINERITC.TTF", None, None)}


def text_of(i):
    return re.sub(r"^\[[A-Z0-9]+\]", "", STR[i], flags=re.M)


def deskew(g, mask, lim=30.0):
    """angle (deg, positive = text runs downhill to the right) maximising the sharpness of the row profile of the printed pixels"""
    h, w = mask.shape
    small = cv2.resize(mask.astype(np.float32), (w // 2, h // 2))
    def score(a):
        M = cv2.getRotationMatrix2D((small.shape[1] / 2, small.shape[0] / 2), a, 1)
        return float(np.var(cv2.warpAffine(small, M, small.shape[::-1]).sum(1)))
    a0 = max(np.arange(-lim, lim + 0.01, 0.5), key=score)
    return max(np.arange(a0 - 0.3, a0 + 0.31, 0.05), key=score)


def ink_mask(rgb, frac=0.65):
    """printed pixels: grey below `frac` of the paper's own grey (the paper is mottled, so a local-contrast test also picks up its stains)"""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return g < frac * np.percentile(g[g.shape[0] // 4:], 60)


def lines_of(mask, y0, y1, gap=3, minw=0):
    out, s = [], None
    prof = mask[y0:y1].sum(1).astype(np.float32)
    rows = prof > max(1.0, 0.04 * float(np.percentile(prof[prof > 0], 75)) if (prof > 0).any() else 1.0)
    for i, v in enumerate(list(rows) + [False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s >= gap:
                xs = np.nonzero(mask[y0 + s:y0 + i].any(0))[0]
                if xs.max() - xs.min() >= minw:
                    out.append((y0 + s, y0 + i, int(xs.min()), int(xs.max())))
            s = None
    return out


def wrap(par, cap):
    words, lines, cur = par.split(), [], ""
    for w in words:
        if len(cur) + (1 if cur else 0) + len(w) <= cap:
            cur = (cur + " " + w) if cur else w
        else:
            lines.append(cur); cur = w
    return lines + [cur]


def wrap_par(p, cap):
    return [l for seg in p.split("\n") for l in wrap(seg, cap)]


def fit_cap(pars, counts):
    best = None
    for cap in range(30, 90):
        n = [len(wrap_par(p, cap)) for p in pars]
        score = sum(abs(a - b) for a, b in zip(n, counts))
        if best is None or score < best[0] or (score == best[0] and cap > best[1]):
            best = (score, cap)
    return best[1]


class Page:
    def __init__(self, img, frac=0.65, region=(0.25, 0.85), angle=None):
        self.img = img
        H, W = img.shape[:2]
        self.H, self.W = H, W
        rgb = img[..., :3]
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        m0 = ink_mask(rgb, frac) & (img[..., 3] > 250)
        m0[:int(H * region[0])] = False; m0[int(H * region[1]):] = False         # the skew of the printed lines, not of the paper's torn edge
        self.angle = deskew(g, m0) if angle is None else angle
        self.M = cv2.getRotationMatrix2D((W / 2, H / 2), self.angle, 1.0)
        self.Mi = cv2.invertAffineTransform(self.M)
        self.rect = cv2.warpAffine(img, self.M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        self.frac = frac
        self.mask = ink_mask(self.rect[..., :3], frac) & (cv2.warpAffine(img[..., 3], self.M, (W, H)) > 250)
        self.region = (int(H * region[0]), int(H * region[1]))

    def lines(self, y0=None, y1=None, gap=3, minw=0):
        return lines_of(self.mask, y0 or self.region[0], y1 or self.region[1], gap, minw)


def render(page, items, size_px, fontkey, ink, jitter=0.35, bold_ranges=None):
    """items: [(text_or_segments, x_left, y_centre)]; returns an (H, W) float layer in the straightened frame. A text may be a list of (string, bold) segments"""
    reg, bold, adv = FONTS[fontkey]
    H, W = page.H, page.W
    layer = Image.new("L", (W * 4, H * 4), 0); dr = ImageDraw.Draw(layer)
    f_reg = ImageFont.truetype(str(reg), int(round(size_px * 4)))
    f_bold = ImageFont.truetype(str(bold), int(round(size_px * 4))) if bold else f_reg
    rng = np.random.default_rng(5)
    for txt, x, yc in items:
        segs = txt if isinstance(txt, list) else [(txt, False)]
        for s, b in segs:
            f = f_bold if b else f_reg
            for ch in s:
                jy = rng.normal(0, jitter); a8 = int(np.clip(rng.normal(235, 14), 170, 255))
                dr.text((x * 4, (yc + jy) * 4), ch, font=f, fill=a8, anchor="lm")
                x += (adv * size_px) if adv else f.getlength(ch) / 4
    la = np.asarray(layer.resize((W, H), Image.LANCZOS), np.float32) / 255
    return np.clip(cv2.GaussianBlur(la, (0, 0), 0.45) * 1.2, 0, 1)


def composite(page, layer, wipe_rows, ink, wipe_x=(30, None), frac_wipe=0.88, rects=None, wipe_mask=None):
    """wipe the old ink of the rows (in the page's own pixels), then lay the straightened text layer on"""
    img, H, W = page.img, page.H, page.W
    y0, y1 = wipe_rows
    x0, x1 = wipe_x[0], (wipe_x[1] or W - 30)
    old = np.zeros((H, W), np.uint8)
    old[y0:y1, x0:x1] = ink_mask(page.rect[..., :3], frac_wipe)[y0:y1, x0:x1] * 255
    if rects:                                                           # only inside these boxes (a signature drawing between two blocks stays)
        keep_ = np.zeros((H, W), np.uint8)
        for (a, b, c, d_) in rects:
            keep_[max(0, b - 10):d_ + 10, max(0, a - 12):c + 12] = 1
        old = old * keep_
    if wipe_mask is not None:
        old = wipe_mask.astype(np.uint8) * 255
    old = cv2.dilate(old, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)) if wipe_mask is None else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    oldp = cv2.warpAffine(old, page.Mi, (W, H), flags=cv2.INTER_NEAREST) > 0
    pg = img[..., :3].astype(np.float32)
    keep = (~oldp & (img[..., 3] > 250)).astype(np.float32)
    smooth = cv2.GaussianBlur(pg * keep[..., None], (0, 0), 7) / np.maximum(cv2.GaussianBlur(keep, (0, 0), 7), 1e-3)[..., None]
    noise = cv2.GaussianBlur(np.random.default_rng(3).normal(0, 1.0, (H, W)).astype(np.float32), (0, 0), 0.8)[..., None] * 2.2
    base = np.where(oldp[..., None], smooth + noise, pg)
    lap = cv2.warpAffine(layer, page.Mi, (W, H), flags=cv2.INTER_CUBIC)[..., None].clip(0, 1)
    out = base * (1 - lap) + np.asarray(ink, np.float32)[None, None, :] * lap
    res = img.copy(); res[..., :3] = out.clip(0, 255).astype(np.uint8)
    return res


def ink_colour(page, rows, darker=0.8):
    y0, y1 = rows
    sel = page.mask[y0:y1] & (cv2.cvtColor(page.rect[y0:y1, :, :3], cv2.COLOR_RGB2GRAY) < 0.5 * np.percentile(cv2.cvtColor(page.rect[..., :3], cv2.COLOR_RGB2GRAY), 60))
    if sel.sum() < 30:
        sel = page.mask[y0:y1]
    return np.median(page.rect[y0:y1, :, :3][sel], axis=0) * darker


def typeset(img, pars, fontkey="COUR", region=(0.25, 0.85), frac=0.65, counts=None, rows=None, cap=None, size=None, darker=0.8, left=None, angle=None, par_gap=1.5, first_split=None, trunc=False, merge=(), cols=None, right=None, wipe_extra=0):
    """pars: paragraph strings (or segment lists); `counts` = lines per paragraph in the old print (found from the gaps when None)"""
    pg = Page(img, frac, region, angle)
    if cols:
        pg.mask[:, :cols[0]] = False; pg.mask[:, cols[1]:] = False
    ls = [l for l in pg.lines(*(rows or (None, None))) if l[1] - l[0] > 5]
    pitch = float(np.median(np.diff([l[0] for l in ls])))
    groups, cur = [], [ls[0]]
    for a, b in zip(ls, ls[1:]):
        if b[0] - a[0] > par_gap * pitch:
            groups.append(cur); cur = []
        cur.append(b)
    groups.append(cur)
    if first_split:                                                     # a heading printed directly above its paragraph
        groups = [groups[0][:first_split], groups[0][first_split:]] + groups[1:]
    for i, j in sorted(merge, reverse=True):                            # blocks the old print ran together
        groups[i:j + 1] = [[l for g in groups[i:j + 1] for l in g]]
    print("angle", round(pg.angle, 2), "blocks", [len(g) for g in groups], "pitch", round(pitch, 1))
    flat = [p if isinstance(p, str) else "".join(s for s, _ in p) for p in pars]
    assert len(flat) == len(groups), (len(flat), [len(g) for g in groups])
    counts = counts or [len(g) for g in groups]
    L = left if left is not None else int(np.median([l[2] for g in groups for l in g]))
    R = right or max(l[3] for g in groups for l in g)
    cap = cap or fit_cap(flat, counts)
    adv = FONTS[fontkey][2] or 0.5
    size = size or (R - L) / (cap * adv) / (1 if fontkey == "COUR" else 1)
    ink = ink_colour(pg, (groups[0][0][0], groups[-1][-1][1]), darker)
    print("cap", cap, "left", L, "right", R, "size", round(size, 1), "ink", ink.round().tolist())
    items = []
    for gi, g in enumerate(groups):
        p = pars[gi]
        if isinstance(p, list):                                         # segments: wrap on the plain text, keep bold flags per character
            plain = "".join(s for s, _ in p); flags = [b for s, b in p for _ in s]
            ls_ = wrap(plain, cap); pos = 0; segl = []
            for ln in ls_:
                while pos < len(plain) and plain[pos] == " " and (pos == 0 or True) and not plain.startswith(ln, pos):
                    pos += 1
                seg = []
                for k, ch in enumerate(ln):
                    b = flags[pos + k] if pos + k < len(flags) else False
                    if seg and seg[-1][1] == b: seg[-1] = (seg[-1][0] + ch, b)
                    else: seg.append((ch, b))
                segl.append(seg); pos += len(ln)
            lines_txt = segl
        else:
            lines_txt = wrap_par(p, cap)
        if len(lines_txt) > len(g) and trunc and gi == len(groups) - 1:     # the page is cut off below: the text runs on out of sight
            lines_txt = lines_txt[:len(g)]
        for li, txt in enumerate(lines_txt):
            if li < len(g):
                yc = (g[li][0] + g[li][1]) / 2
            else:
                yc = (g[-1][0] + g[-1][1]) / 2 + (li - len(g) + 1) * pitch
            items.append((txt, L, yc))
    layer = render(pg, items, size * 1.0, fontkey, ink)
    wr = (groups[0][0][0] - 8, groups[-1][-1][1] + 8 + wipe_extra + int(pitch * max(0, max(len(wrap_par(f, cap)) - c for f, c in zip(flat, counts)))))
    return composite(pg, layer, wr, ink)


def run(name, src, dst):
    im = np.asarray(Image.open(src).convert("RGBA"))
    Image.fromarray(ITEMS[name](im)).save(dst)


def doc5(im):
    a = text_of(40020).replace(" \n", " ")
    b = text_of(40021)
    pars = [a] + [re.sub(r"\s+", " ", p.replace("\n", " ")) for p in b.split("\n\n")[:4]] + ["With respect,", "King Makani'Ehe"]
    out = typeset(im, pars, "COUR", region=(0.25, 0.82), frac=0.65, first_split=1)
    return out


def paras(*ids):
    t = "\n".join(text_of(i) for i in ids)
    return [re.sub(r"\s+", " ", p.replace("\n", " ")).strip() for p in re.split(r"\n\s*\n", t) if p.strip()]


def doc3(im):
    """the Kestrel Publishing letter: date, salutation, four paragraphs (the thumbnail runs the last two together)"""
    P = paras(40016, 40017)
    body = [P[1].split("Dear Mr. Zachary,", 1)[1].strip(), P[2], P[3], P[4] + " " + P[5]]
    return typeset(im, [P[0], "Dear Mr. Zachary,"] + body, "COUR", region=(0.17, 0.72), frac=0.65, par_gap=1.4)


def doc2(im):
    """Addendum to Articles of Piracy: the thumbnail runs the first refit into the lead-in, and the page is cut off below the third refit"""
    P = paras(40009)
    pars = [P[0], P[1] + " " + P[2], " ".join(P[3:])]
    return typeset(im, pars, "COUR", region=(0.20, 0.97), frac=0.6, par_gap=1.08, trunc=True, darker=0.7, merge=[(2, 3)], cap=70)


def telegram1(im):
    """Crawford's telegram (typed capitals): the two address lines, the 7-line message, the signature"""
    t = text_of(40125).replace("<B>", "").replace("<b>", "")
    body = re.sub(r"\s+", " ", t.split("\n\n", 1)[1]).strip()
    pars = ["TO-NATHAN ZACHARY\nFROM-MAJ. LOYLE CRAWFORD BROADWAY BOMBERS", body, "MAJOR LOYLE CRAWFORD"]
    return typeset(im, pars, "COUR", region=(0.235, 0.85), frac=0.6, par_gap=1.8, darker=0.8)


def telegram2(im):
    """Tennyson's telegram: only the typed message (six lines); the printed form and the handwriting stay"""
    t = text_of(40126).replace("<B>", "").replace("<b>", "")
    body = re.sub(r"\s+", " ", t.split("\n\n", 1)[1]).strip()
    return typeset(im, [body], "COUR", region=(0.36, 0.66), frac=0.6, par_gap=1.8, darker=0.8)


def conletter(im):
    """Midtown Construction letter: address block, salutation, one paragraph that runs off the bottom of the (folded) page"""
    t = text_of(40096)
    head, rest = t.split("\n\nDear Mr. Zachary,\n\n", 1)
    return typeset(im, [head, "Dear Mr. Zachary,", re.sub(r"\s+", " ", rest).strip()], "COUR", region=(0.25, 0.93), frac=0.6, par_gap=1.5, trunc=True, cols=(90, 1010), right=950, wipe_extra=40)


ITEMS = {"doc5": doc5, "doc3": doc3, "doc2": doc2, "telegram1": telegram1, "telegram2": telegram2, "conletter": conletter}

if __name__ == "__main__":
    if sys.argv[1] == "analyze":
        im = np.asarray(Image.open(sys.argv[3]).convert("RGBA"))
        pg = Page(im, float(sys.argv[6]) if len(sys.argv) > 6 else 0.65)
        y0, y1 = (int(float(sys.argv[4]) * pg.H), int(float(sys.argv[5]) * pg.H)) if len(sys.argv) > 5 else (None, None)
        print("angle", pg.angle)
        for l in pg.lines(y0, y1):
            print(l)
    else:
        run(sys.argv[2], sys.argv[3], sys.argv[4])


# ---------------------------------------------------------------- thumbnails whose baked text is not the string's own wrapping
import difflib, subprocess, tempfile, os


def ocr_line(img_rgb):
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "l.png"; cv2.imwrite(str(p), g)
        t = subprocess.run(["tesseract", str(p), "stdout", "--psm", "7"], capture_output=True, text=True,
                           env=dict(os.environ, TESSDATA_PREFIX=str(Path.home() / ".local/share/tessdata"))).stdout
    return " ".join(t.split())


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def align(ocr_lines, game):
    """for every OCR line the slice of `game` it most likely prints: the OCR text is matched against the game text (difflib blocks); a line's slice runs from where its
    first matched character falls in the game text to where the next line's first matched character falls (word-snapped), so the slices tile the text"""
    g_idx = [i for i, c in enumerate(game) if re.match(r"[A-Za-z0-9]", c)]
    g_norm = "".join(game[i].lower() for i in g_idx)
    o_norm, o_start = "", []
    for ln in ocr_lines:
        o_start.append(len(o_norm)); o_norm += norm(ln)
    sm = difflib.SequenceMatcher(None, o_norm, g_norm, autojunk=False)
    pairs = []
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            pairs.append((a + k, b + k))
    def at(pos):                                                        # game offset of OCR offset `pos` (nearest matched character at or after it)
        for a, b in pairs:
            if a >= pos:
                return b
        return len(g_norm)
    starts = [at(s) for s in o_start]
    starts = np.maximum.accumulate(starts).tolist()
    cut = []
    for s in starts:
        k = g_idx[min(s, len(g_idx) - 1)]
        while k > 0 and game[k - 1] not in " \n":
            k -= 1
        cut.append(k)
    cut.append(len(game))
    return [game[cut[i]:cut[i + 1]].strip() for i in range(len(ocr_lines))]


def typeset_aligned(shipped, orig, game, fontkey="COUR", region=(0.2, 0.9), frac=0.65, size=None, darker=0.8, angle=None, minw=0, bold_regex=None, keep_para=True, dbg=None):
    """shipped = the redraw's thumbnail (RGBA); orig = the original thumbnail (RGBA, any size); game = the game's text for it"""
    pg = Page(shipped, frac, region, angle)
    H, W = pg.H, pg.W
    o = cv2.resize(np.asarray(orig, np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
    o_rect = cv2.warpAffine(o, pg.M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    ls = [l for l in pg.lines(minw=minw) if l[1] - l[0] > 5]
    reads = []
    for (t, b, x0, x1) in ls:
        pad = max(2, (b - t) // 6)
        reads.append(ocr_line(o_rect[max(0, t - pad):b + pad, max(0, x0 - pad):x1 + pad, :3]))
    sl = align(reads, game)
    if dbg:
        for l, r, s in zip(ls, reads, sl):
            print(l, "|", r, "|", s)
    adv = FONTS[fontkey][2] or 0.5
    # one font size for the page: from the lines whose slice is long enough to measure
    est = [(x1 - x0) / (len(s) * adv) for (t, b, x0, x1), s in zip(ls, sl) if len(s) > 25]
    size = size or float(np.median(est))
    ink = ink_colour(pg, (ls[0][0], ls[-1][1]), darker)
    items = []
    for (t, b, x0, x1), s in zip(ls, sl):
        if not s:
            continue
        sz = size
        if len(s) * adv * sz > (x1 - x0) * 1.12:
            sz = (x1 - x0) * 1.12 / (len(s) * adv)
        items.append((s, x0, (t + b) / 2, sz))
    layer = None
    Himg = Image.new("L", (W * 4, H * 4), 0); dr = ImageDraw.Draw(Himg); rng = np.random.default_rng(5)
    reg, bold, _ = FONTS[fontkey]
    for s, x, yc, sz in items:
        f = ImageFont.truetype(str(reg), int(round(sz * 4)))
        for ch in s:
            jy = rng.normal(0, 0.35); a8 = int(np.clip(rng.normal(235, 14), 170, 255))
            dr.text((x * 4, (yc + jy) * 4), ch, font=f, fill=a8, anchor="lm")
            x += (adv * sz) if adv else f.getlength(ch) / 4
    la = np.asarray(Himg.resize((W, H), Image.LANCZOS), np.float32) / 255
    layer = np.clip(cv2.GaussianBlur(la, (0, 0), 0.45) * 1.2, 0, 1)
    return composite(pg, layer, (ls[0][0] - 8, ls[-1][1] + 8), ink)


def column_lines(pg, x0, x1, y0, y1, gap=2):
    sub = pg.mask[:, x0:x1]
    return [(t, b, l + x0, r + x0) for (t, b, l, r) in lines_of(sub, y0, y1, gap)]


# ---------------------------------------------------------------- handwritten notes: the game's own handwriting fonts, fitted into the blocks the old print occupied
def font_of(key, px):
    return ImageFont.truetype(str(FONTS[key][0]), int(round(px)))


def wrap_px(text, font, width):
    out = []
    for seg in text.split("\n"):
        cur = ""
        for w in seg.split():
            t = (cur + " " + w) if cur else w
            if font.getlength(t) <= width or not cur:
                cur = t
            else:
                out.append(cur); cur = w
        out.append(cur)
    return out


def blocks_of(pg, y0, y1, min_gap=14, pct=1.0, min_h=8):
    """text blocks of the straightened page: runs of rows with ink, separated by at least min_gap empty rows -> (x0, y0, x1, y1) by ink percentiles"""
    m = pg.mask[y0:y1]
    prof = m.sum(1)
    on = prof > 1
    blocks, s, gap = [], None, 0
    for i, v in enumerate(list(on) + [False] * (min_gap + 1)):
        if v:
            if s is None: s = i
            gap = 0
        elif s is not None:
            gap += 1
            if gap >= min_gap:
                e = i - gap + 1
                if e - s >= min_h:
                    sub = m[s:e]
                    xs = np.nonzero(sub.any(0))[0]
                    cx = np.nonzero(sub.sum(0) > 0)[0]
                    blocks.append((int(np.percentile(cx, pct)), y0 + s, int(np.percentile(cx, 100 - pct)), y0 + e))
                s = None
    return blocks


def fit_block(text, fontkey, box, lh_extra=1.0):
    """largest font size at which `text` wrapped to the box width still fits the box height (a one-line block: the size at which it spans the width)"""
    x0, y0, x1, y1 = box
    W, Hh = x1 - x0, y1 - y0
    best = 8
    for s in range(8, 120):
        f = font_of(fontkey, s)
        asc, desc = f.getmetrics()
        lines = wrap_px(text, f, W)
        h = len(lines) * (asc + desc) * lh_extra
        if h <= Hh * 1.08 and max(f.getlength(l) for l in lines) <= W * 1.06:
            best = s
    return best


def typeset_blocks(im, pars, fontkey, region=(0.1, 0.95), frac=0.62, min_gap=14, darker=0.9, angle=None, boxes=None, sizes=None, col=None, dbg=True, lh_extra=1.0, x_shift=0):
    pg = Page(im, frac, region, angle)
    if col:
        pg.mask[:, :col[0]] = False; pg.mask[:, col[1]:] = False
    bl = boxes or blocks_of(pg, pg.region[0], pg.region[1], min_gap)
    if dbg:
        print("angle", round(pg.angle, 2), "blocks", bl)
    assert len(bl) == len(pars), (len(bl), len(pars))
    ink = ink_colour(pg, (bl[0][1], bl[-1][3]), darker)
    H, W = pg.H, pg.W
    layer = Image.new("L", (W * 4, H * 4), 0); dr = ImageDraw.Draw(layer)
    rng = np.random.default_rng(7)
    fits = [fit_block(t, fontkey, b, lh_extra) for b, t in zip(bl, pars)]
    multi = [f for f, b, t in zip(fits, bl, pars) if len(wrap_px(t, font_of(fontkey, f), b[2] - b[0])) >= 3]
    body = int(min(multi)) if multi else None                      # one pen: the paragraphs share a size, a heading or signature may be up to 1.3x
    for i, (box, text) in enumerate(zip(bl, pars)):
        s = (sizes or {}).get(i) or (fits[i] if body is None else (body if fits[i] <= body * 1.3 and len(wrap_px(text, font_of(fontkey, fits[i]), box[2] - box[0])) >= 2 else min(fits[i], int(body * 1.3))))
        f4 = font_of(fontkey, s * 4); f = font_of(fontkey, s)
        asc, desc = f.getmetrics()
        lines = wrap_px(text, f, box[2] - box[0])
        if dbg:
            print(i, "size", s, "lines", len(lines), repr(lines[0][:40]))
        y = box[1] + asc
        for ln in lines:
            dr.text(((box[0] + x_shift) * 4, y * 4), ln, font=f4, fill=235, anchor="ls")
            y += (asc + desc) * lh_extra
    la = np.asarray(layer.resize((W, H), Image.LANCZOS), np.float32) / 255
    layer = np.clip(cv2.GaussianBlur(la, (0, 0), 0.5) * 1.15, 0, 1)
    wipe = (bl[0][1] - 10, bl[-1][3] + 10)
    return composite(pg, layer, wipe, ink, wipe_x=(20, None), frac_wipe=min(0.9, frac + 0.2), rects=bl)
