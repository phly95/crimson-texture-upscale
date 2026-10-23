"""mag_sb_00_00_mag1.py [OUT_DIR]: the Air Action Weekly cover (SB_00_00_MAG1) with its baked lettering put right.
Klein redrew the small print badly: the tagline, MAGAZINE, the date, No.13 with its two marks, the imprint and the price line under the rule, and
the two cover-line blocks. Each is wiped (its own ink, the paper filled from the paper round it) and re-set on the straightened 4x picture in a
condensed bold face stretched to the old print's cap height and span; the cover lines are set in a level frame and rotated back (their tilt is
17.6 deg). The masthead, WEEKLY, the 25 cents badge, the seal, the signature and the photograph are not touched (the signature's "a" in Zachary is
still a "w": a cursive letter joined to the h and r over the hand, not repaired here). The tagline, cover lines and date are the printed wording
(the game has no string for them); the imprint and price line are inferred from the faint print (see the report).
Writes OUT_DIR/SB_00_00_MAG1_4x_edit.png (straightened) and OUT_DIR/SB_00_00_MAG1_page.png (page frame, via tile_mags_rot.back).
Usage: venv/bin/python -I tools/mag_sb_00_00_mag1.py [OUT_DIR]
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tile_mags_rot as tm

STEM = "SB_00_00_MAG1"
DEFAULT_OUT = Path("/tmp/claude-1000/-home-deck/8d33977a-9b65-41a7-af0e-5ee75413ed5e/scratchpad/mags_text") / STEM
DJ = Path("/usr/share/fonts/TTF")
F_COND = DJ / "DejaVuSansCondensed-Bold.ttf"              # tagline, MAGAZINE, date, price, imprint, No.13
F_OBL = DJ / "DejaVuSansCondensed-BoldOblique.ttf"        # cover-line heads and captions
F_OBL_LT = DJ / "DejaVuSansCondensed-Oblique.ttf"         # "Interview with" (lighter)
ANGLE = 17.6                                              # the cover lines rise 17.6 deg to the right in the straightened picture

CREAM = np.array([224, 196, 166], np.float32)             # the tagline's cream (sampled)
DARK = np.array([46, 13, 16], np.float32)                 # MAGAZINE (sampled)
RED_DATE = np.array([181, 19, 33], np.float32)            # the date (sampled)
RED_PRICE = np.array([162, 25, 28], np.float32)           # the price line (sampled)
IMPRINT_INK = np.array([50, 8, 10], np.float32)
NO13_INK = np.array([33, 6, 8], np.float32)
SHADOW = np.array([25, 20, 22], np.float32)
STRIP = (20, 10, 22)                                      # the black strip under the rule (flat)


def text_mask(text, font, cap, x0, x1, base, shape):
    """coverage of `text` in a level frame of `shape`: capitals `cap` px high, baseline at row `base`, ink spanning columns x0..x1 (the glyphs
    are stretched to that span, so the face is squeezed to the old print's width)"""
    f200 = ImageFont.truetype(str(font), 200)
    b = f200.getbbox("H")
    size = int(round(200.0 * cap / (b[3] - b[1])))
    f = ImageFont.truetype(str(font), size)
    Wc, Hc = int(f.getlength(text)) + 60, int(size * 2.5)
    yb = int(size * 1.6)
    im = Image.new("L", (Wc, Hc), 0)
    ImageDraw.Draw(im).text((30, yb), text, font=f, fill=255, anchor="ls")
    a = np.asarray(im, np.float32) / 255
    cols = np.nonzero(a.max(0) > 0.25)[0]
    c0, c1 = cols.min(), cols.max() + 1
    sx = (x1 - x0) / float(c1 - c0)
    M = np.float32([[sx, 0, x0 - c0 * sx], [0, 1, base - yb]])
    out = cv2.warpAffine(a, M, (shape[1], shape[0]), flags=cv2.INTER_LINEAR)
    return np.clip(cv2.GaussianBlur(out, (0, 0), 0.5), 0, 1)


def fill_paper(img, wipe, seed):
    """the paper under a wiped area: smooth fill from the pixels round it (multi-scale normalised blur), plus grain matched to the paper's own"""
    rgb = img.astype(np.float32)
    keep = (~wipe).astype(np.float32)
    out = rgb * keep[..., None]
    filled = keep.copy()
    for sg in (3, 8, 24, 80):
        num = cv2.GaussianBlur(rgb * keep[..., None], (0, 0), sg)
        den = cv2.GaussianBlur(keep, (0, 0), sg)
        new = (filled < 0.5) & (den > 0.02)
        out[new] = (num / np.maximum(den, 1e-6)[..., None])[new]
        filled[new] = 1
    ring = cv2.dilate(wipe.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ring &= keep > 0.5
    smooth_ring = out[ring]
    resid = rgb[ring] - smooth_ring
    grain = float(1.4826 * np.median(np.abs(resid - np.median(resid)))) * 0.7 if resid.size else 1.5
    noise = np.random.default_rng(seed).normal(0, grain, wipe.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.5)[..., None]
    return np.where(wipe[..., None], np.clip(out + noise, 0, 255), rgb)


def paint(img, a, ink, shadow=None):
    """lay ink (coverage a, 0..1) on the picture; the optional shadow (alpha, colour) goes under it"""
    out = img.astype(np.float32)
    if shadow is not None:
        out = out * (1 - shadow[0][..., None]) + shadow[1][None, None, :] * shadow[0][..., None]
    out = out * (1 - a[..., None]) + np.asarray(ink, np.float32)[None, None, :] * a[..., None]
    return out


def rect(shape, x0, x1, y0, y1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def level_text(img, items):
    """items: (text, font, cap, x0, x1, base, ink, wipe_rect[, flat]); the wipes are done first (the paper filled from the paper round them; a
    flat colour for the black strip, which the red rule above would otherwise tint), then the ink on top in one pass"""
    H, W = img.shape[:2]
    wipe = np.zeros((H, W), bool)
    flats = []
    Lum = 0.3 * img[..., 0] + 0.59 * img[..., 1] + 0.11 * img[..., 2]
    for it in items:
        m = rect((H, W), *it[7])
        if len(it) > 9 and it[9] is not None:                       # only the old ink inside the box (dark below the threshold), grown a little
            m = cv2.dilate((m & (Lum < it[9][1])).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3) > 0
        if len(it) > 8 and it[8] is not None:
            flats.append((m, np.asarray(it[8], np.float32)))
        else:
            wipe |= m
    cur = fill_paper(img, wipe, seed=3)
    g = cv2.GaussianBlur(np.random.default_rng(9).normal(0, 6.0, (H, W)).astype(np.float32), (0, 0), 0.5)[..., None]
    for m, col in flats:
        cur = np.where(m[..., None], np.clip(col[None, None, :] + g, 0, 255), cur)
    for it in items:
        text, font, cap, x0, x1, base, ink = it[:7]
        a = text_mask(text, font, cap, x0, x1, base, (H, W))
        cur = paint(cur, a, ink)
    return cur, wipe


# --- the level (straightened) lines: the tagline, MAGAZINE, the date, the price line, the imprint, No.13 (ink rows measured on the 4x picture)
LEVEL = [
    # the tagline: cap 35, the old span 218..1372, baseline 228.8 (the wipe covers the old letters and their descenders)
    ("North America's #1 Publication Covering Topics Important to Aviators and Aviation.", F_COND, 35, 218, 1372, 228.8, CREAM, (212, 1378, 186, 238)),
    ("MAGAZINE", F_COND, 26, 766, 888, 560.5, DARK, (760, 894, 530, 564)),
    ("March 17, 1939", F_COND, 30, 1178, 1437, 638.5, RED_DATE, (1168, 1447, 605, 643), STRIP),
    ("ES, DC, PC 45 cents – ME, TX, OK 35 cents", F_COND, 18, 456, 868, 626.5, RED_PRICE, (452, 876, 603, 633), STRIP),   # right of the badge
    ("Reg. Canadian Patent Office – Printed in Canada", F_COND, 10, 620, 1028, 589.5, IMPRINT_INK, (612, 1036, 575, 589)),
    ("o.13", F_COND, 32, 242, 316, 497.5, NO13_INK, (230, 322, 462, 499), None, ("dark", 60)),
]


def stub_no13(img):
    """the torn cover's N: only the top of its right stem shows at the edge of the cover (a short dark tick left of the o)"""
    a = np.zeros(img.shape[:2], np.float32)
    a[464:475, 236:240] = 0.85
    return paint(img, a, NO13_INK)


def marks_from_original(img, orig_src):
    """the two small dark strokes under No.13: too small to read in the original (two short strokes, 1x x 62-63 and x 65, rows 129-131), so they
    are copied from its own darkness (absolute threshold, the paper there is 55-70) and laid at 4x; the redraw's marks there are wiped first"""
    H, W = img.shape[:2]
    Lum = 0.3 * img[..., 0] + 0.59 * img[..., 1] + 0.11 * img[..., 2]
    box = rect((H, W), 226, 268, 506, 540)
    wipe = cv2.dilate((box & (Lum < 60)).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3) > 0   # the redraw's own dark marks only (the torn edge stays)
    cur = fill_paper(img, wipe, seed=7)
    x0, x1, y0, y1 = 61, 66, 128, 133
    L = 0.3 * orig_src[y0:y1, x0:x1, 0] + 0.59 * orig_src[y0:y1, x0:x1, 1] + 0.11 * orig_src[y0:y1, x0:x1, 2]
    d = np.clip((58.0 - L.astype(np.float32)) / 28.0, 0, 1)
    d4 = cv2.resize(d, (d.shape[1] * 4, d.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
    a = np.zeros((H, W), np.float32)
    a[4 * y0:4 * y0 + d4.shape[0], 4 * x0:4 * x0 + d4.shape[1]] = np.clip(cv2.GaussianBlur(d4, (0, 0), 0.6), 0, 1)
    return paint(cur, a, NO13_INK), a


# --- the cover lines: set in a level frame (their own baselines and spans, measured on the picture), rotated back into the straightened picture
H_LINES = [
    # text, font, cap, x0, x1, baseline (level frame), wipe box (x0, x1, y0, y1)
    ("Exclusive!", F_OBL, 34, 445, 602, 677, (453, 800, 634, 686)),
    ("Interview with", F_OBL_LT, 27, 607, 795, 677, (453, 800, 634, 686)),
    ("NATHAN ZACHARY", F_OBL, 28, 512, 775, 720, (504, 783, 684, 728)),
    ("SEE PAGE 3", F_OBL, 27, 590, 757, 762, (582, 765, 726, 770)),
    ("Secret Photos!", F_OBL, 36, 500, 737, 856, (492, 745, 812, 864)),
    ("Inside a PIRATE Zeppelin", F_OBL, 26, 355, 715, 900, (347, 723, 866, 908)),
    ("SEE PAGE 6", F_OBL, 27, 535, 702, 944, (527, 710, 912, 952)),
]


def cover_lines(img, log):
    """the old cover lines are wiped where their own ink is (cream letters and their dark shadow, against the photo's grey), the paper under
    them filled from the grey round them; the new lines are set in the level frame and only the changed pixels go back into the picture"""
    H, W = img.shape[:2]
    c = (W / 2.0, H / 2.0)
    M_neg = cv2.getRotationMatrix2D(c, -ANGLE, 1.0)               # straightened -> level
    M_pos = cv2.getRotationMatrix2D(c, ANGLE, 1.0)                # level -> straightened
    hh = cv2.warpAffine(img, M_neg, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    L = 0.3 * hh[..., 0] + 0.59 * hh[..., 1] + 0.11 * hh[..., 2]
    ink = np.zeros((H, W), bool)
    text_h = np.zeros((H, W), np.float32)
    shadow_h = np.zeros((H, W), np.float32)
    for text, font, cap, x0, x1, base, (wx0, wx1, wy0, wy1) in H_LINES:
        box = np.zeros((H, W), bool)
        box[wy0:wy1, wx0:wx1] = True
        P = float(np.median(L[box]))
        ink |= box & ((L > P + 35) | (L < P - 30))                  # cream letters and dark shadow only
        a = text_mask(text, font, cap, x0, x1, base, (H, W))
        text_h = np.maximum(text_h, a)
        sh = np.zeros((H, W), np.float32)
        sh[3:, 3:] = a[:-3, :-3]
        shadow_h = np.maximum(shadow_h, cv2.GaussianBlur(sh, (0, 0), 1.0))
    ink = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    keep = (~ink).astype(np.float32)
    out = hh * keep[..., None]
    filled = keep.copy()
    for sg in (2, 5, 12, 30):                                       # the paper under the old ink, from the grey round it
        num = cv2.GaussianBlur(hh * keep[..., None], (0, 0), sg)
        den = cv2.GaussianBlur(keep, (0, 0), sg)
        new = (filled < 0.5) & (den > 0.02)
        out[new] = (num / np.maximum(den, 1e-6)[..., None])[new]
        filled[new] = 1
    grain = 2.0
    noise = cv2.GaussianBlur(np.random.default_rng(5).normal(0, grain, (H, W)).astype(np.float32), (0, 0), 0.5)[..., None]
    out = np.where(ink[..., None], out + noise, hh)
    cream = np.array([236, 226, 206], np.float32)
    out = paint(out, text_h, cream, shadow=(np.clip(shadow_h, 0, 1) * 0.85, SHADOW))
    changed_h = ink | (text_h > 0.01) | (shadow_h > 0.01)
    back = cv2.warpAffine(out, M_pos, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cu = np.clip(cv2.warpAffine(changed_h.astype(np.float32), M_pos, (W, H), flags=cv2.INTER_LINEAR), 0, 1)[..., None]
    log.append(("cover lines: %d lines re-set" % len(H_LINES), int((cu[..., 0] > 0.01).sum())))
    return img.astype(np.float32) * (1 - cu) + back * cu


def build(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hd = np.asarray(Image.open(ROOT / "tile_job" / f"rot_{STEM}_4x.png").convert("RGB")).astype(np.float32)
    orig_src = np.asarray(Image.open(ROOT / "tile_job" / f"rot_{STEM}_src.png").convert("RGB"))
    log = []
    cur, wipe = level_text(hd, LEVEL)
    cur = stub_no13(cur)
    cur, _ = marks_from_original(cur, orig_src)
    log.append(("level lines re-set", int(wipe.sum())))
    cur = cover_lines(cur, log)
    edit = np.clip(np.rint(cur), 0, 255).astype(np.uint8)
    ed = out_dir / f"{STEM}_4x_edit.png"
    Image.fromarray(edit).save(ed)
    page = tm.back(STEM, src=ed, dst=out_dir / f"{STEM}_page.png")
    for row in log:
        print(row)
    print("edit", ed, "page", page)
    return ed, page


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
