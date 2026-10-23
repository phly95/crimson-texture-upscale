"""restore_aaw.py IN.png OUT.png --orig ORIGINAL.png [--debug DIR]: the Air Action Weekly cover (SB_00_00_MAG1) with its baked-in lettering put right.
Klein redraws the small lettering of the cover badly. The true strings are read from the original picture (the masthead is only ~8 px there, so
each glyph was checked at 10x against the letters around it). A letter that came out wrong is replaced by the same letter taken from another place
on the same cover (same font, size and baseline), so nothing has to be invented; the fine-print imprint under MAGAZINE is wiped and re-set.
The work is done on the straightened picture (tagline baseline) and only the changed pixels are rotated back into the page, so the rest of the
cover (alpha, photo, rules, price badge, date) keeps the input's own pixels.
Usage: venv/bin/python -I tools/restore_aaw.py IN.png OUT.png --orig ORIGINAL.png [--debug DIR]
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ANGLE = -7.2                       # the tagline's baseline (fitted on the letter bottoms)
ROWS = (44, 80)                    # the tagline band (the cover's top edge line lies above it)


def straighten(img, angle=ANGLE):
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), M


def soft_cream(S):
    """0..1 cream coverage from the green channel: red paper is ~20-40, cream lettering ~200-250. The transparent border of the cover is stored
    green (R ~ 0), so pixels with no red are not paper and not lettering."""
    G = S[..., 1].astype(np.float32)
    R = S[..., 0].astype(np.float32)
    return np.clip((G - 45.0) / 155.0, 0, 1) * (R > 120)


def band_paper(S, A):
    """smooth red paper colour under the band (normalised convolution over its non-cream pixels) and the grain of that paper"""
    H, W = A.shape
    keep = np.zeros((H, W), np.float32)
    keep[ROWS[0]:ROWS[1]] = (A[ROWS[0]:ROWS[1]] < 0.02) & (S[ROWS[0]:ROWS[1], :, 0] > 120)
    rgb = S[..., :3].astype(np.float32)
    num = cv2.GaussianBlur(rgb * keep[..., None], (0, 0), 2.5)
    den = cv2.GaussianBlur(keep, (0, 0), 2.5)[..., None]
    smooth = num / np.maximum(den, 1e-3)
    grain = float(np.std((rgb - smooth)[keep > 0]))
    return smooth, grain


def cream_colour(S, A):
    band = np.zeros(A.shape, bool)
    band[ROWS[0]:ROWS[1]] = True
    R, G, B = (S[..., i].astype(np.int32) for i in range(3))
    sel = band & (A > 0.9) & (R > 200) & (G > 190) & (B > 170)
    return np.median(S[..., :3][sel].astype(np.float32), axis=0)


def move_glyph(S, A, P, grain, src, dst, dy, F, seed, src_rows=ROWS):
    """wipe the glyph in columns dst=(d0, d1) and lay the glyph in columns src=(s0, s1) in its place: a translation along the baseline (dy for a
    different height), centred on the slot. A = cream coverage of the input (the source and the old glyph come from it), P = paper, F = cream colour.
    Returns (picture, changed mask)."""
    H, W = A.shape
    r0, r1 = ROWS
    s0, s1 = src
    d0, d1 = dst
    src_a = np.zeros((H, W), np.float32)
    src_a[src_rows[0]:src_rows[1], s0:s1] = A[src_rows[0]:src_rows[1], s0:s1]
    old = np.zeros((H, W), np.float32)
    old[r0:r1, d0:d1] = A[r0:r1, d0:d1]
    wipe = cv2.dilate((old > 0.02).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    dx = ((d0 + d1) - (s0 + s1)) / 2.0
    new_a = cv2.warpAffine(src_a, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), flags=cv2.INTER_LINEAR)
    new_a = np.clip(new_a, 0, 1)
    changed = wipe | (new_a > 0.01)
    rng = np.random.default_rng(seed)
    fill = P + rng.normal(0, grain, (H, W)).astype(np.float32)[..., None]
    a = new_a[..., None]
    out = S.copy()
    region = changed[..., None]
    val = fill * (1 - a) + F[None, None, :] * a
    out[..., :3] = np.where(region, np.clip(np.rint(val), 0, 255), S[..., :3]).astype(np.uint8)
    return out, changed


# the letter swaps, as (what is wrong, columns of the wrong glyph, columns of the right glyph), straightened picture, one baseline.
# Every column range is bounded by paper columns, so the glyphs next to it are not touched.
TAG_FIXES = [
    ("America: n -> r", (151, 160), (513, 521)),        # 'r' from Important
    ("America: e -> c", (164, 173), (277, 284)),        # 'c' from Publication
    ("Publication: b -> t", (295, 302), (560, 566)),    # 't' from 'to'
    ("Covering: w -> e", (362, 370), (142, 150)),       # 'e' from America
    ("Aviators: i -> t", (618, 625), (560, 566)),       # 't' from 'to'
    ("Important: a -> o", (503, 512), (566, 576)),      # 'o' from 'to'
    ("Important: l -> t", (548, 553), (560, 566)),      # 't' from 'to'
    ("Aviation: b -> t", (730, 739), (560, 566)),       # 't' from 'to'
]
STOP = ("Aviation: full stop", (764, 768), (160, 164), 16.0)   # the i-dot of America dropped to the baseline after the n (x 753-762)


def fix_tagline(S, log):
    A = soft_cream(S)
    P, grain = band_paper(S, A)
    F = cream_colour(S, A)
    changed_all = np.zeros(A.shape, bool)
    cur = S
    for k, (name, dst, src) in enumerate(TAG_FIXES):
        cur, ch = move_glyph(cur, A, P, grain, src, dst, 0.0, F, seed=k)
        changed_all |= ch
        log.append((name, int(ch.sum())))
    name, dst, src, dy = STOP
    cur, ch = move_glyph(cur, A, P, grain, src, dst, dy, F, seed=99, src_rows=(46, 56))
    changed_all |= ch
    log.append((name, int(ch.sum())))
    return cur, changed_all


# the imprint under MAGAZINE: printed in the same dark ink as MAGAZINE, a few pixels high, centred over the cover (x 303-548) above the rule
IMPRINT = "Reg. Canadian Patent Office \u2013 Printed in Canada"
IMPRINT_X = (303, 548)             # the span of the printed line in the straightened picture
IMPRINT_BASE = 292.5               # its baseline (the rule is at y 293)
IMPRINT_ROWS = (279, 292)          # rows that may hold the old imprint: below MAGAZINE, above the rule
IMPRINT_COLS = (296, 556)


def set_imprint(S, log):
    H, W = S.shape[:2]
    r0, r1 = IMPRINT_ROWS
    c0, c1 = IMPRINT_COLS
    rgb = S[..., :3].astype(np.float32)
    L = 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
    paper = float(np.median(L[r0:r1, c0:c1]))
    ink_mask = np.zeros((H, W), bool)
    ink_mask[r0:r1 + 1, c0:c1] = L[r0:r1 + 1, c0:c1] < paper - 35
    ink_mask = cv2.dilate(ink_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ink_mask[r1 + 1:] = False
    keep = np.zeros((H, W), np.float32)
    keep[r0 - 6:r1 + 1, c0 - 6:c1 + 6] = 1.0
    keep[ink_mask] = 0.0
    num = cv2.GaussianBlur(rgb * keep[..., None], (0, 0), 1.5)
    den = cv2.GaussianBlur(keep, (0, 0), 1.5)[..., None]
    smooth = num / np.maximum(den, 1e-3)
    rng = np.random.default_rng(7)
    grain = float(np.std((rgb - smooth)[(keep > 0) & (np.arange(H)[:, None] >= r0) & (np.arange(H)[:, None] <= r1)]))
    fill = smooth + rng.normal(0, grain, (H, W)).astype(np.float32)[..., None]
    ink_rgb = np.array([22.0, 2.0, 2.0], np.float32)                   # the dark ink of MAGAZINE above it
    # the line: the original's imprint is a ~5 px condensed print, letter-spaced out to IMPRINT_X; set char by char at 4x with the tracking that fills the span
    font_path = "/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf"
    cap = 5.4
    f0 = ImageFont.truetype(font_path, 100)
    hb = f0.getbbox("H")
    size = 100 * cap * 4 / (hb[3] - hb[1])
    f = ImageFont.truetype(font_path, int(round(size)))
    span = IMPRINT_X[1] - IMPRINT_X[0]
    widths = [f.getlength(c) for c in IMPRINT]
    track = (span * 4 - sum(widths)) / (len(IMPRINT) - 1)
    cw, ch = int(span + 16), 24
    canvas = Image.new("L", (cw * 4, ch * 4), 0)
    d = ImageDraw.Draw(canvas)
    x = 8 * 4.0
    for c, w in zip(IMPRINT, widths):
        d.text((x, (ch / 2) * 4), c, font=f, fill=255, anchor="ls")
        x += w + track
    small = np.clip(np.asarray(canvas.resize((cw, ch), Image.LANCZOS), np.float32) / 255.0 * 1.25, 0, 1)
    a = np.zeros((H, W), np.float32)
    y0 = int(round(IMPRINT_BASE - ch / 2 - 1))
    x0 = IMPRINT_X[0] - 8
    a[y0:y0 + ch, x0:x0 + cw] = small
    a = np.clip(a, 0, 1)
    changed = ink_mask | (a > 0.01)
    val = fill * (1 - a[..., None]) + ink_rgb[None, None, :] * a[..., None]
    out = S.copy()
    out[..., :3] = np.where(changed[..., None], np.clip(np.rint(val), 0, 255), S[..., :3]).astype(np.uint8)
    log.append(("imprint: re-set %r" % IMPRINT, int(changed.sum()), round(size, 2)))
    return out, changed


# the Secret block ("Inside a PIRATE Zeppelin / SEE PAGE 6"): Klein drew the Z of SEE as a Z and the 6 as a C. The original's S and 6 are on
# the same line, so they are taken from the original picture (already at the redraw's size and straightened the same way), sharpened and set
# in the block's cream with its dark shadow.
GLYPH_FIXES = [
    ("SEE: Z -> S", (251, 266, 546, 571), (252, 268, 545, 571)),     # original S box, redraw Z box (x0, x1, y0, y1)
    ("PAGE 6: C -> 6", (336, 355, 516, 545), (338, 358, 515, 546)),  # original 6 box, redraw C box
]


def glyph_alpha(img, box):
    """coverage of a cream glyph in a box of a straightened picture: luminance between the local paper and the cream, sharpened"""
    x0, x1, y0, y1 = box
    rgb = img[y0:y1, x0:x1, :3].astype(np.float32)
    L = 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
    lo, hi = np.percentile(L, 20), np.percentile(L, 97)
    a = np.clip((L - lo) / max(hi - lo, 1.0), 0, 1)
    t = np.clip((a - 0.3) / 0.45, 0, 1)
    return t * t * (3 - 2 * t)


def paste_orig_glyph(S, SO, obox, hbox, log, name, dst_mask=None):
    H, W = S.shape[:2]
    x0, x1, y0, y1 = hbox
    rgb = S[..., :3].astype(np.float32)
    L_all = 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
    # the redraw's glyph inside its box: cream coverage (against this block's brown paper) and centroid
    hb = np.zeros((H, W), np.float32)
    hb[y0:y1, x0:x1] = glyph_alpha(S, hbox)
    if dst_mask is not None:
        hb = hb * dst_mask.astype(np.float32)
    ys, xs = np.mgrid[0:H, 0:W]
    m_h = hb.sum()
    cxh, cyh = (hb * xs).sum() / m_h, (hb * ys).sum() / m_h
    # the original's glyph (sharpened) and its centroid
    ao = glyph_alpha(SO, obox)
    ox0, ox1, oy0, oy1 = obox
    oc = np.zeros((H, W), np.float32)
    oc[oy0:oy1, ox0:ox1] = ao
    m_o = oc.sum()
    cxo, cyo = (oc * xs).sum() / m_o, (oc * ys).sum() / m_o
    dx, dy = cxh - cxo, cyh - cyo
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    gl = cv2.warpAffine(oc, M, (W, H), flags=cv2.INTER_LINEAR)
    sh = cv2.warpAffine(oc, np.float32([[1, 0, dx + 2.0], [0, 1, dy + 2.0]]), (W, H), flags=cv2.INTER_LINEAR)
    sh = cv2.GaussianBlur(sh, (0, 0), 0.6) * 0.9
    # wipe the redraw's glyph and its dark shadow in the box, put the local paper back
    near = cv2.dilate((hb > 0.1).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    shadow_px = near & (L_all < 60)
    wipe = np.zeros((H, W), bool)
    wipe[y0:y1, x0:x1] = (hb[y0:y1, x0:x1] > 0.1)
    wipe |= shadow_px
    wipe = cv2.dilate(wipe.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    wipe[:y0 - 2] = False; wipe[y1 + 2:] = False; wipe[:, :x0 - 2] = False; wipe[:, x1 + 2:] = False
    if dst_mask is not None:
        wipe = dst_mask.copy()
    keep = np.zeros((H, W), np.float32)
    keep[y0 - 6:y1 + 6, x0 - 6:x1 + 6] = 1
    keep[wipe] = 0
    smooth = cv2.GaussianBlur(rgb * keep[..., None], (0, 0), 2.0) / np.maximum(cv2.GaussianBlur(keep, (0, 0), 2.0), 1e-3)[..., None]
    resid = (rgb - smooth)[(keep > 0) & (~wipe)]
    grain = float(1.4826 * np.median(np.abs(resid - np.median(resid)))) * 0.6 if resid.size else 2.0
    rng = np.random.default_rng(11)
    fill = smooth + rng.normal(0, grain, (H, W)).astype(np.float32)[..., None]
    # the block's cream and the shadow colour, from the redraw's own letters of the same block
    cream = np.percentile(rgb[(hb > 0.5)], 80, axis=0) if (hb > 0.5).any() else np.array([220.0, 232.0, 240.0])
    shadow_rgb = np.array([22.0, 20.0, 24.0], np.float32)
    a_sh = sh[..., None]
    a_g = gl[..., None]
    val = fill * (1 - a_sh) + shadow_rgb * a_sh
    val = val * (1 - a_g) + cream[None, None, :] * a_g
    changed = wipe | (gl > 0.01) | (sh > 0.01)
    out = S.copy()
    out[..., :3] = np.where(changed[..., None], np.clip(np.rint(val), 0, 255), S[..., :3]).astype(np.uint8)
    log.append((name, int(changed.sum()), [round(float(dx), 2), round(float(dy), 2)], [round(float(v), 0) for v in cream]))
    return out, changed


# the signature "Nathan Zachary: Gentleman Pirate!": the a after the N came out as a grey blob with specks (the stray dots next to the N), and two grey
# smudges sit beside the h and under the next a. The blob is replaced by the clean a of the same word (same line, its shadow carried over) and the
# smudges are taken off the paper; the N and the letters around them are not touched.
def light_ink(S, box, P):
    """cream plus dark shadow coverage of a box: 0..1, relative to the local paper luminance P"""
    x0, x1, y0, y1 = box
    H, W = S.shape[:2]
    a_c = np.zeros((H, W), np.float32)
    a_c[y0:y1, x0:x1] = glyph_alpha(S, box)
    L = 0.3 * S[..., 0].astype(np.float32) + 0.59 * S[..., 1].astype(np.float32) + 0.11 * S[..., 2].astype(np.float32)
    near = cv2.dilate((a_c > 0.05).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    a_d = np.clip((P - L) / max(P - 40.0, 1.0), 0, 1) * near
    a_d[:y0] = 0; a_d[y1:] = 0; a_d[:, :x0] = 0; a_d[:, x1:] = 0
    return np.clip(a_c + a_d, 0, 1)


def paper_L(S, box, a):
    """median paper luminance in a ring around the box (pixels with no glyph coverage a)"""
    x0, x1, y0, y1 = box
    L = 0.3 * S[..., 0].astype(np.float32) + 0.59 * S[..., 1].astype(np.float32) + 0.11 * S[..., 2].astype(np.float32)
    sel = np.zeros(L.shape, bool)
    sel[y0 - 6:y1 + 6, x0 - 6:x1 + 6] = True
    sel &= a < 0.02
    return float(np.median(L[sel])) if sel.any() else 120.0


def fill_paper(S, wipe, log, name, seed=21, grain=None):
    H, W = S.shape[:2]
    rgb = S[..., :3].astype(np.float32)
    keep = (~wipe).astype(np.float32)
    smooth = cv2.GaussianBlur(rgb * keep[..., None], (0, 0), 2.0) / np.maximum(cv2.GaussianBlur(keep, (0, 0), 2.0), 1e-3)[..., None]
    ring = cv2.dilate(wipe.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    resid = (rgb - smooth)[ring & (~wipe)]
    if grain is None:
        grain = float(1.4826 * np.median(np.abs(resid - np.median(resid)))) * 0.6 if resid.size else 2.0
    rng = np.random.default_rng(seed)
    fill = smooth + rng.normal(0, grain, (H, W)).astype(np.float32)[..., None]
    out = S.copy()
    out[..., :3] = np.where(wipe[..., None], np.clip(np.rint(fill), 0, 255), S[..., :3]).astype(np.uint8)
    log.append((name, int(wipe.sum())))
    return out


def erase_specks(S, box, lo, hi, log, name):
    """grey smudge pixels inside a box where no lettering is: pixels lighter than the paper by lo..hi (fraction of the way to the cream)"""
    H, W = S.shape[:2]
    rgb = S[..., :3].astype(np.float32)
    L = 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
    x0, x1, y0, y1 = box
    P = float(np.median(L[y0 - 4:y1 + 4, x0 - 4:x1 + 4]))
    cream_L = float(np.percentile(L[y0:y1, x0:x1], 99))
    sel = np.zeros((H, W), bool)
    band = (L > P + lo * (cream_L - P)) & (L < P + hi * (cream_L - P))
    sel[y0:y1, x0:x1] = band[y0:y1, x0:x1]
    sel = cv2.dilate(sel.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    sel[:y0] = False; sel[y1:] = False; sel[:, :x0] = False; sel[:, x1:] = False
    out = fill_paper(S, sel, log, name, grain=1.5)
    return out, sel


SIG_BLOB = (117, 138, 927, 950)       # the grey blob that replaces the a (its own connected component is taken)
SIG_A_SRC = (169, 189, 922, 953)      # the clean a of Nathan (x 170-187), on the same line
SIG_SMUDGE = [(189, 200, 940, 952), (155, 164, 923, 936)]   # the grey smudges beside the h and under the next a


def fix_signature(S, log):
    """the grey blob takes the clean a of the same word (Nathan, x 170-187): laid with whole-pixel shifts so the thin script strokes stay sharp;
    its horizontal centre on the blob's, its baseline (bottom of the bowl) on the blob's bottom"""
    H, W = S.shape[:2]
    x0, x1, y0, y1 = SIG_BLOB
    a = light_ink(S, SIG_BLOB, paper_L(S, SIG_BLOB, light_ink(S, SIG_BLOB, 120.0)))
    cream = np.zeros((H, W), np.uint8)
    cream[y0:y1, x0:x1] = (a[y0:y1, x0:x1] > 0.5)
    n, lab, st, cen = cv2.connectedComponentsWithStats(cream, 8)
    blob_lab = None
    for i in range(1, n):
        cx, cy = cen[i]
        if st[i, 4] >= 20 and x0 <= cx < x1 and y0 <= cy < y1:
            blob_lab = i if blob_lab is None or st[i, 4] > st[blob_lab, 4] else blob_lab
    letters = (cream > 0) & (lab != blob_lab)
    letters = cv2.dilate(letters.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    blob = np.zeros((H, W), bool)
    blob[lab == blob_lab] = True
    halo = np.zeros((H, W), bool)
    halo[y0:y1, x0:x1] = a[y0:y1, x0:x1] > 0.12
    wipe = (cv2.dilate(blob.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) | halo
    wipe &= ~letters
    wipe[:y0] = False; wipe[y1:] = False; wipe[:, :x0] = False; wipe[:, x1:] = False
    # the clean a: its cream coverage and shadow, moved by whole pixels
    sx0, sx1, sy0, sy1 = SIG_A_SRC
    P = paper_L(S, SIG_A_SRC, light_ink(S, SIG_A_SRC, 120.0))
    a_src = light_ink(S, SIG_A_SRC, P)
    cream_src = glyph_alpha(S, SIG_A_SRC)
    ys, xs = np.mgrid[0:H, 0:W]
    cs = np.zeros((H, W), np.float32)
    cs[sy0:sy1, sx0:sx1] = cream_src
    cx_s = (cs * xs).sum() / cs.sum()
    rows_s = np.nonzero((cs > 0.5).any(1))[0]
    bottom_s = rows_s.max()
    bm = np.zeros((H, W), np.float32)
    bm[blob] = 1.0
    cx_d = (bm * xs).sum() / bm.sum()
    rows_d = np.nonzero((blob).any(1))[0]
    bottom_d = rows_d.max()
    dx = int(round(cx_d - cx_s))
    dy = int(round(bottom_d - bottom_s))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    a_new = cv2.warpAffine(a_src, M, (W, H), flags=cv2.INTER_NEAREST)
    rgb = S[..., :3].astype(np.float32)
    fg_src = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_NEAREST)
    fill_mask = wipe.copy()
    out = fill_paper(S, fill_mask, log, "Nathan: blob wiped", grain=1.5)
    fill = out[..., :3].astype(np.float32)
    region = (a_new > 0.01) | fill_mask
    val = fill * (1 - a_new[..., None]) + fg_src * a_new[..., None]
    out[..., :3] = np.where(region[..., None], np.clip(np.rint(val), 0, 255), out[..., :3]).astype(np.uint8)
    log.append(("Nathan: a blob -> clean a", int(region.sum()), [dx, dy]))
    for k, box2 in enumerate(SIG_SMUDGE):
        out, sel = erase_specks(out, box2, 0.2, 0.75, log, "Nathan: smudge %d" % k)
        region = region | sel
    return out, region


# the two marks under "No.13": the original has two dark strokes there (too small to read); the redraw turned them into an O and an l. They are put back
# as the original draws them: the redraw's dark strokes wiped, the original's strokes laid on the same paper.
MARKS_BOX = (70, 86, 244, 260)      # inside the torn edge of the cover (its left edge is grey in the original)


def fix_marks(S, SO, log):
    H, W = S.shape[:2]
    x0, x1, y0, y1 = MARKS_BOX
    rgb = S[..., :3].astype(np.float32)
    L = 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
    Lo = 0.3 * SO[..., 0].astype(np.float32) + 0.59 * SO[..., 1].astype(np.float32) + 0.11 * SO[..., 2].astype(np.float32)
    P = float(np.median(L[y0:y1, x0:x1]))
    P_o = float(np.median(Lo[y0:y1, x0:x1]))
    dark = np.zeros((H, W), bool)
    dark[y0:y1, x0:x1] = L[y0:y1, x0:x1] < P - 40
    wipe = cv2.dilate(dark.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    a = np.zeros((H, W), np.float32)
    a_o = np.clip((P_o - Lo[y0:y1, x0:x1]) / max(P_o - 25.0, 1.0), 0, 1)
    t = np.clip((a_o - 0.35) / 0.5, 0, 1)
    a[y0:y1, x0:x1] = t * t * (3 - 2 * t)
    out = fill_paper(S, wipe, log, "No.13 marks: redraw strokes wiped", grain=1.5)
    fill = out[..., :3].astype(np.float32)
    region = wipe | (a > 0.01)
    val = fill * (1 - a[..., None]) + np.array([22.0, 2.0, 2.0], np.float32)[None, None, :] * a[..., None]
    out[..., :3] = np.where(region[..., None], np.clip(np.rint(val), 0, 255), out[..., :3]).astype(np.uint8)
    log.append(("No.13 marks: original strokes laid", int(region.sum())))
    return out, region


def main():
    args = sys.argv[1:]
    orig = None
    debug = None
    if "--orig" in args:
        k = args.index("--orig"); orig = args[k + 1]; del args[k:k + 2]
    if "--debug" in args:
        k = args.index("--debug"); debug = Path(args[k + 1]); del args[k:k + 2]
    if len(args) != 2 or orig is None:
        raise SystemExit("usage: restore_aaw.py IN.png OUT.png --orig ORIGINAL.png [--debug DIR]")
    src, dst = args
    im = np.asarray(Image.open(src).convert("RGBA"))
    H, W = im.shape[:2]
    S, M = straighten(im)
    SO = straighten(np.asarray(Image.open(orig).convert("RGBA").resize((W, H), Image.LANCZOS)))[0]
    log = []
    S2, changed = fix_tagline(S, log)
    S2, ch2 = set_imprint(S2, log)
    changed = changed | ch2
    for name, obox, hbox in GLYPH_FIXES:
        S2, ch3 = paste_orig_glyph(S2, SO, obox, hbox, log, name)
        changed = changed | ch3
    S2, ch4 = fix_signature(S2, log)
    changed = changed | ch4
    S2, ch5 = fix_marks(S2, SO, log)
    changed = changed | ch5
    for row in log:
        print(row)
    # back into the page: only the changed pixels are rotated back, everything else is the input's own pixel
    Mi = cv2.invertAffineTransform(M)
    m = np.clip(cv2.warpAffine(changed.astype(np.float32), Mi, (W, H), flags=cv2.INTER_LINEAR), 0, 1)
    back = cv2.warpAffine(S2[..., :3].astype(np.float32), Mi, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    res = im.copy()
    blend = im[..., :3].astype(np.float32) * (1 - m[..., None]) + back * m[..., None]
    sel = m > 0.01
    res[..., :3][sel] = np.clip(np.rint(blend[sel]), 0, 255).astype(np.uint8)
    outside = im[..., 3] == 0                                  # the transparent border keeps the input's own colour
    res[..., :3][outside] = im[..., :3][outside]
    Image.fromarray(res, "RGBA").save(dst)
    print("changed pixels", int(np.any(res != im, axis=2).sum()))
    if debug is not None:
        debug.mkdir(parents=True, exist_ok=True)
        Image.fromarray(S2[..., :3]).save(debug / "straight_after.png")
        Image.fromarray(SO[..., :3]).save(debug / "orig_straight.png")


if __name__ == "__main__":
    main()
