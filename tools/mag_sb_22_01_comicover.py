"""mag_sb_22_01_comicover.py [OUT_PAGE.png]: the lettering of the 'NATHAN ZACHARY: Fortune Hunter!' comic cover (SB_22_01_COMICOVER) in the straightened frame.
  - the left slogan (GUTS OF STEEL, / NERVES OF STEEL, / HEART OF GOLD): the garbled lines are wiped and set again in Arial Bold on the printed baselines and right ends
    (the paper edge cuts the first letters, as in the print);
  - the right blurb (NATHAN ZACHARY TAKES ON THE / BLACK HATS and THE BLACK HATS / THE BLACK SWAN, WHO GETS WHO? / FIND OUT!!!!): wiped where it sits on paper and set
    in Arial Bold on the printed baselines, the first words in orange;
  - the price badge ('3' and cent sign, the diagonal band with its logo): the redraw's badge is a plain rectangle reading '34' and its band is gone, so the badge area is taken
    from the original, upscaled to 4x, and feathered into the redraw;
  - the title: the redraw sits about 7 px (1x) higher than the print, so the top of 'NATHAN' falls outside the cover's own edge and is cut. fix_title moves the rows
    smoothly down (30 px at 4x from row 140 to 290, tapering to 0 by row 420), which also squashes 'ZACHARY' a little toward its printed height; the rails stay straight.
Writes the edited straightened copy to the scratch folder and the page-frame RGBA (tile_mags_rot.back).
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tile_mags_rot as tm

STEM = "SB_22_01_COMICOVER"
SRC = ROOT / "tile_job" / f"rot_{STEM}_4x.png"
SCR = Path("/tmp/claude-1000/-home-deck/8d33977a-9b65-41a7-af0e-5ee75413ed5e/scratchpad/mags_text") / STEM
FONT = Path.home() / ".local/share/fonts/winfonts/arialbd.ttf"
INK = (34, 30, 28)                                      # the blurb's black ink
SLOGAN_INK = (58, 46, 38)                               # the slogan's brown ink
ORANGE = (224, 94, 44)                                  # 'NATHAN ZACHARY' in the blurb
SLOGAN_PX = 18.0                                        # 4x px: cap height ~13 px, as printed
BLURB_PX = 17.5
# slogan lines: text, right end of the ink, baseline (4x grid)
SLOGAN = [("GUTS OF STEEL,", 590.0, 350.0), ("NERVES OF STEEL,", 612.0, 377.0), ("HEART OF GOLD", 597.0, 403.0)]
# blurb lines: list of (text, colour); left edge and baseline (4x grid)
BLURB_X = 1119.5
BLURB_L2_WIDTH = 313.5                                  # the printed ink width of 'BLACK HATS and THE BLACK HATS' (4x px)
BLURB = [([("NATHAN ZACHARY", ORANGE), (" TAKES ON THE", INK)], 521.0),
         ([("BLACK HATS and THE BLACK HATS", INK)], 547.0),
         ([("THE BLACK SWAN, WHO GETS WHO?", INK)], 573.5),
         ([("FIND OUT!!!!", INK)], 599.0)]


def orig_up4():
    """the original, straightened exactly as the redraw's canvas is, at 4x (8x Lanczos, then area-averaged down to 4x)"""
    im = Image.open(tm.SB / (STEM + ".PNG")).convert("RGBA")
    W, H = im.size
    M, (W2, H2) = tm.forward(W, H, tm.ANGLE[STEM])
    k = 8
    S8 = np.diag([k, k, 1.0])
    M8 = (S8 @ np.vstack([M, [0, 0, 1]]) @ np.linalg.inv(S8))[:2]
    big = np.asarray(im.resize((W * k, H * k), Image.LANCZOS)).astype(np.float32)
    a = big[..., 3:4] / 255
    rgb = big[..., :3] * a + 128 * (1 - a)                # the transparent part is grey, as in the other tools
    up = cv2.warpAffine(rgb, M8, (W2 * k, H2 * k), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
    return cv2.resize(up, (W2 * 4, H2 * 4), interpolation=cv2.INTER_AREA).clip(0, 255).astype(np.uint8)


def hsv_of(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.int32)
    return hsv[..., 0], hsv[..., 1], hsv[..., 2]


def line_alpha(shape, text, px, baseline, x_origin=None, x_right=None, sx=1.0):
    """coverage (4x grid, full canvas `shape`) of one line of Arial Bold: its origin at x_origin, or its ink ending at x_right; sits on the baseline.
    sx squeezes the glyphs horizontally (the print is narrower than Arial at the same cap height)"""
    H, W = shape
    f = ImageFont.truetype(str(FONT), int(round(px * 4)))
    l, t, r, b = f.getbbox(text, anchor="ls")
    ox = x_origin if x_origin is not None else x_right - sx * r / 4
    x0, y0 = int(np.floor(ox + sx * l / 4)) - 4, int(np.floor(baseline + t / 4)) - 4
    w, h = int(np.ceil(ox + sx * r / 4)) - x0 + 8, int(np.ceil(baseline + b / 4)) - y0 + 8
    im = Image.new("L", (int(np.ceil(w * 4 / sx)), h * 4), 0)
    ImageDraw.Draw(im).text(((ox - x0) * 4 / sx, (baseline - y0) * 4), text, font=f, fill=255, anchor="ls")
    im = im.resize((w * 4, h * 4), Image.LANCZOS)
    patch = np.asarray(im.resize((w, h), Image.BOX), np.float32) / 255
    full = np.zeros((H, W), np.float32)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    full[ya:yb, xa:xb] = patch[ya - y0:yb - y0, xa - x0:xb - x0]
    return full


def paint_colour(img, alpha, colour, soft=0.45):
    a = cv2.GaussianBlur(alpha, (0, 0), soft)[..., None]
    return (img.astype(np.float32) * (1 - a) + np.float32(colour) * a).clip(0, 255).astype(np.uint8)


def fix_slogan(img):
    """the three slogan lines: the garbled ink (dark, on paper, left of the Z) is inpainted, the printed lines are laid on"""
    H, W = img.shape[:2]
    h, s, v = hsv_of(img)
    box = np.zeros((H, W), bool)
    box[316:432, 440:624] = True                        # the garbled slogan has moved down with the title too (the Z's outline is kept by the orange test)
    dark = (v < 140) & box
    orange = cv2.dilate((((h >= 2) & (h <= 18) & (s >= 120) & (v >= 120)).astype(np.uint8)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    old = dark & ~orange
    old = cv2.dilate(old.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    out = cv2.inpaint(img, old * 255, 3, cv2.INPAINT_TELEA)
    alpha = np.zeros((H, W), np.float32)
    for text, xr, base in SLOGAN:
        alpha = np.maximum(alpha, line_alpha((H, W), text, SLOGAN_PX, base, x_right=xr))
    print("slogan: old ink px", int(old.sum()), "new ink px", int((alpha > 0.5).sum()))
    return paint_colour(out, alpha, SLOGAN_INK, soft=0.6)     # the print is a brown, softer ink than the blurb's


def paper_fill(img, hole, keep, sigma=5.0):
    """the hole's pixels := the paper round it, a normalised-convolution average of the kept paper pixels (no darker halo is carried in)"""
    k = keep.astype(np.float32)
    f = img.astype(np.float32)
    num = cv2.GaussianBlur(f * k[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(k, (0, 0), sigma)
    est = num / np.maximum(den, 1e-4)[..., None]
    out = img.copy()
    ok = hole & (den > 0.05)
    out[ok] = est[ok].clip(0, 255).astype(np.uint8)
    rest = hole & ~ok
    if rest.any():
        out = cv2.inpaint(out, rest.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    return out


def fix_blurb(img):
    """the blurb: its ink is wiped where it sits on paper (the statue's spikes and outlines stay), then the printed lines are laid on.
    One horizontal squeeze (from the printed width of the second line) is used for all four lines."""
    H, W = img.shape[:2]
    h, s, v = hsv_of(img)
    box = np.zeros((H, W), bool)
    box[494:606, 1110:1472] = True
    paper = (s < 90) & (v > 165)
    paper_frac = cv2.blur(paper.astype(np.float32), (11, 11))
    # ink = anything clearly darker or more coloured than the paper round it (the faint old orange and grey too), not the green of the statue
    local = cv2.blur(img.astype(np.float32), (15, 15))
    diff = np.abs(img.astype(np.float32) - local).sum(-1)
    green = (h >= 35) & (h <= 90) & (s >= 60)
    text = (((diff > 30) | (v < 150)) & box & (paper_frac > 0.3) & ~green) | (((v < 150) & (s < 110)) & box & (paper_frac > 0.55))
    top = np.zeros((H, W), bool)                        # the old orange 'NATHAN ZACHARY' leaves bright specks above the new first line
    top[490:512, 1110:1330] = True
    text |= ((h <= 20) & (s >= 90) & (v >= 110)) & top
    old = cv2.dilate(text.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))   # the halos of the old ink go too
    out = paper_fill(img, old > 0, (v > 110) & (s < 110) & ~green & (old == 0), sigma=7.0)     # the paper and its shading, not the ink
    f = ImageFont.truetype(str(FONT), int(round(BLURB_PX * 4)))
    l2, _, r2, _ = f.getbbox(BLURB[1][0][0][0], anchor="ls")
    sx = BLURB_L2_WIDTH / ((r2 - l2) / 4)
    print("blurb: squeeze", round(sx, 3), "old ink px", int(old.sum()))
    for segs, base in BLURB:
        x = BLURB_X - sx * f.getbbox(segs[0][0], anchor="ls")[0] / 4
        for seg, colour in segs:
            a = line_alpha((H, W), seg, BLURB_PX, base, x_origin=x, sx=sx)
            out = paint_colour(out, a, colour)
            x += sx * f.getlength(seg) / 4
    return out


def fix_badge(img):
    """the badge area (the '34' rectangle and the empty corner) := the original's print, upscaled, feathered into the redraw"""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = 458, 166, 586, 338                 # below the paper's top edge; the redraw's badge has moved down with the title
    orig = orig_up4()
    w = np.zeros((H, W), np.float32)
    w[y0:y1, x0:x1] = 1
    f = 6
    w = cv2.GaussianBlur(w, (0, 0), f / 2.5)
    w[y0 + f:y1 - f, x0 + f:x1 - f] = np.maximum(w[y0 + f:y1 - f, x0 + f:x1 - f], 1)
    w = np.clip(w, 0, 1)[..., None]
    return (img.astype(np.float32) * (1 - w) + orig.astype(np.float32) * w).clip(0, 255).astype(np.uint8)


def fix_title(img):
    """the printed title sits lower than the redraw's (about 30 px at 4x): rows are moved down by a smooth amount, 30 px from row 140 to row 290, tapering to 12 px
    at row 405 and to 0 by row 420 (the glove under the Z stays put). 'NATHAN' moves down whole, 'ZACHARY' is squashed a little toward its printed height. Rows are
    remapped smoothly, so the straight rails stay straight, and no letter is erased (nothing ghosts)."""
    H, W = img.shape[:2]
    ys = np.arange(H, dtype=np.float32)
    d = np.interp(ys, [0, 140, 290, 405, 420], [0, 30, 30, 12, 0]).astype(np.float32)
    map_y = np.repeat((ys - d)[:, None], W, axis=1)                  # output row y takes its colour from row y - d(y)
    map_x = np.repeat(np.arange(W, dtype=np.float32)[None, :], H, axis=0)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def run(src=SRC):
    img = np.asarray(Image.open(src).convert("RGB")).copy()
    img = fix_title(img)
    img = fix_badge(img)
    img = fix_slogan(img)
    img = fix_blurb(img)
    return img


def changed_boxes(a, b, thr=3):
    diff = (np.abs(a[..., :3].astype(int) - b[..., :3].astype(int)).max(-1) > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(diff, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))), connectivity=8)
    return [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, _ in st[1:]]


if __name__ == "__main__":
    SCR.mkdir(parents=True, exist_ok=True)
    edited = SCR / f"{STEM}_text_4x.png"
    Image.fromarray(run()).save(edited)
    dst = Path(sys.argv[1]) if len(sys.argv) > 1 else SCR / f"{STEM}_text_page.png"
    tm.back(STEM, src=edited, dst=dst)
    new = np.asarray(Image.open(dst).convert("RGBA"))
    old = np.asarray(Image.open(ROOT / "tile_job" / f"rot_{STEM}_page.png").convert("RGBA"))
    assert new.shape == old.shape and np.array_equal(new[..., 3], old[..., 3]), "alpha must match the original's"
    print("page", dst, new.shape)
    print("changed boxes (page px, 4x):", changed_boxes(new, old))
