"""mag_sb_19_01_zpjcover.py [OUT_PAGE.png]: the lettering of the 'Zeppelin Pilot's Journal' cover (SB_19_01_ZPJCOVER, PNG variant) in the straightened frame.
The tiled redraw garbles four pieces of print. Each is wiped (only its own ink, found by colour, inpainted) and set again:
  - the diagonal date on the black banner, 'August 2, 1937': Arial Bold squeezed to the original's ink box, set in the banner's frame (+41 degrees) and turned back;
  - the two small gold lines under the headline, '"The Empire State-Texas Zeppelin Race Heats Up!"' and '"Imperial Air Opens New South America Route"': Arial Bold at the
    size and place fitted to the original's gold ink (one size fits both lines), with the headline's soft drop shadow;
  - the red note 'Mado' -> 'Made': the 'o' is replaced by the 'e' of 'the' (same handwriting) moved along the note's tilted baseline.
The masthead, headline, price, tab and the photograph are left alone. Writes the edited straightened copy to the scratch folder and the page-frame RGBA (tile_mags_rot.back).
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tile_mags_rot as tm

STEM = "SB_19_01_ZPJCOVER"
SRC = ROOT / "tile_job" / f"rot_{STEM}_4x.png"
SCR = Path("/tmp/claude-1000/-home-deck/8d33977a-9b65-41a7-af0e-5ee75413ed5e/scratchpad/mags_text") / STEM
ARIAL_B = Path.home() / ".local/share/fonts/winfonts/arialbd.ttf"
C4 = (1217.5, 212.5)                                    # the banner's rotation centre, 4x upright canvas
ANG = 41.0                                              # the banner's diagonal: the date runs down-right at +41 degrees
M_UP2BAN = cv2.getRotationMatrix2D(C4, ANG, 1.0)        # upright -> banner frame (date horizontal)
M_BAN2UP = cv2.getRotationMatrix2D(C4, -ANG, 1.0)
GOLD_SMALL = (222, 170, 36)                             # the cover's gold (median of the printed gold lines)
DATE_GOLD = (224, 172, 30)
DATE_BOX = (1118.0, 195.5, 200.0, 36.5)                 # the date's ink box in the banner frame, 4x: left, top, width, height (cap top to the g descender)
FONT_PX = 30.95                                         # 4x px; fitted to both lines' ink widths (234..983, 244..941)
LINES = [  # text, ink left, baseline (4x), wipe band (y0, y1, x0, x1)
    ('"The Empire State-Texas Zeppelin Race Heats Up!"', 234.0, 908.0, (866, 930, 220, 1000)),   # the headline's descenders end at 860
    ('"Imperial Air Opens New South America Route"', 244.0, 1021.0, (985, 1046, 230, 960)),
]
SHADOW = (30, 24, 18)
NOTE_O_BOX = (411, 1177, 448, 1226)                     # the 'o' of 'Mado' (x0, y0, x1, y1)
NOTE_E_BOX = (539, 1156, 580, 1210)                     # the 'e' of 'the' (x0, y0, x1, y1), clear of its neighbours
NOTE_SHIFT = (-130, 18)                                 # 'e' -> 'o' position: along the note's baseline (tilt about -8 degrees)
KERNEL = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def shift(a, dx, dy):
    return cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR)


def ink_alpha(text, px, box, shape):
    """coverage (4x grid, full canvas `shape`) of `text` in Arial Bold at `px`; its ink is squeezed to box = (left, top, width, height)"""
    H, W = shape
    f = ImageFont.truetype(str(ARIAL_B), int(round(px * 4)))
    l, t, r, b = f.getbbox(text, anchor="ls")                     # supersampled ink box relative to the origin on the baseline
    pad = 8
    im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - l, pad - t), text, font=f, fill=255, anchor="ls")
    x0, y0, w, h = box
    px0, py0 = int(np.floor(x0)) - 3, int(np.floor(y0)) - 3       # patch origin on the 4x grid
    pw, ph = int(np.ceil(x0 + w)) - px0 + 3, int(np.ceil(y0 + h)) - py0 + 3
    sx, sy = 4 * w / (r - l), 4 * h / (b - t)
    Mss = np.float32([[sx, 0, 4 * (x0 - px0) - sx * pad], [0, sy, 4 * (y0 - py0) - sy * pad]])
    ss = cv2.warpAffine(np.asarray(im, np.float32), Mss, (4 * pw, 4 * ph), flags=cv2.INTER_LINEAR)
    full = np.zeros((H, W), np.float32)
    full[py0:py0 + ph, px0:px0 + pw] = cv2.resize(ss, (pw, ph), interpolation=cv2.INTER_AREA) / 255
    return full


def line_box(text, xl, yb):
    """the ink box a line set at FONT_PX occupies when its ink starts at xl and sits on the baseline yb (4x grid)"""
    l, t, r, b = ImageFont.truetype(str(ARIAL_B), int(round(FONT_PX * 4))).getbbox(text, anchor="ls")
    return (xl, yb + t / 4, (r - l) / 4, (b - t) / 4)


def wipe(img, mask):
    """inpaint the old ink (mask) from its surroundings"""
    return cv2.inpaint(img, mask.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)


def paint(img, alpha, colour, shadow=None, soft=0.45):
    """the text's coverage laid on in its ink colour (softened a little), with an optional shadow under it"""
    a = cv2.GaussianBlur(alpha, (0, 0), soft)[..., None]
    out = img.astype(np.float32)
    if shadow is not None:
        s = shadow[..., None]
        out = out * (1 - s) + np.float32(SHADOW) * s
    out = out * (1 - a) + np.float32(colour) * a
    return out.clip(0, 255).astype(np.uint8)


def fix_date(img):
    """'August 2, 1937' on the banner: the smear is found in the banner frame (the stripe is black there, the thin cracks are dropped by the opening)"""
    H, W = img.shape[:2]
    ban = cv2.warpAffine(img, M_UP2BAN, (W, H), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
    L = cv2.cvtColor(ban, cv2.COLOR_RGB2GRAY).astype(np.float32)
    black = float(np.median(L[190:240, 1040:1100]))
    rect = np.zeros((H, W), bool)
    rect[182:246, 1100:1340] = True                        # below the stripe's top edge and its crack
    smear = ((cv2.GaussianBlur(L, (0, 0), 1.5) > black + 28) & rect).astype(np.uint8)
    smear = cv2.morphologyEx(smear, cv2.MORPH_OPEN, KERNEL(5))   # letter strokes survive, the thin crack lines do not
    smear = cv2.dilate(smear, KERNEL(5))
    smear_up = cv2.warpAffine(smear * 255, M_BAN2UP, (W, H), flags=cv2.INTER_NEAREST) > 127
    img = wipe(img, smear_up)
    alpha = ink_alpha("August 2, 1937", 36.0, DATE_BOX, (H, W))
    alpha_up = cv2.warpAffine(alpha, M_BAN2UP, (W, H), flags=cv2.INTER_LINEAR)
    print("date: black", round(black, 1), "smear px", int(smear.sum()))
    return paint(img, alpha_up, DATE_GOLD, soft=0.7)           # the original's date is a blurred, distressed print


def fix_lines(img):
    """the two small gold lines: old gold ink plus the dark shadow next to it is wiped, the new lines and their shadow are laid on"""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gold = (h >= 6) & (h <= 28) & (s >= 100) & (v >= 110)
    dark = v < 90
    old = np.zeros((H, W), bool)
    alpha = np.zeros((H, W), np.float32)
    for text, xl, yb, (y0, y1, x0, x1) in LINES:
        band = np.zeros((H, W), bool)
        band[y0:y1, x0:x1] = True
        g = gold & band
        near = cv2.dilate(g.astype(np.uint8), KERNEL(9)) > 0
        old |= cv2.dilate((g | (near & dark & band)).astype(np.uint8), KERNEL(3)) > 0
        alpha = np.maximum(alpha, ink_alpha(text, FONT_PX, line_box(text, xl, yb), (H, W)))
    shadow = cv2.GaussianBlur(shift(alpha, 3, 4), (0, 0), 1.6) * 0.6
    print("lines: old ink px", int(old.sum()), "new ink px", int((alpha > 0.5).sum()))
    return paint(wipe(img, old), alpha, GOLD_SMALL, shadow)


def fix_note(img):
    """'Mado' -> 'Made': the red 'o' component is inpainted, and the 'e' of 'the' is laid in its place along the baseline"""
    H, W = img.shape[:2]
    f = img.astype(np.float32)
    redn = f[..., 0] - np.maximum(f[..., 1], f[..., 2])
    ared = np.clip((redn - 45) / 80, 0, 1)
    n, lab = cv2.connectedComponents((ared > 0.5).astype(np.uint8), connectivity=8)
    x0o, y0o, x1o, y1o = NOTE_O_BOX
    ids, cnt = np.unique(lab[y0o:y1o, x0o:x1o][lab[y0o:y1o, x0o:x1o] > 0], return_counts=True)
    assert len(ids), "no red ink in the 'o' box"
    o_lab = ids[np.argmax(cnt)]                         # the 'o' stroke (its hole is background, so no seed point)
    o_core = cv2.dilate((lab == o_lab).astype(np.uint8), KERNEL(5)) > 0
    fringe = np.zeros_like(o_core)
    fringe[y0o - 3:y1o + 3, x0o - 3:x1o + 3] = True
    o_mask = (ared > 0.12) & fringe & cv2.dilate(o_core.astype(np.uint8), KERNEL(7)).astype(bool)   # the stroke and its faint red edge only
    img = cv2.inpaint(img, o_mask.astype(np.uint8) * 255, 5, cv2.INPAINT_NS)
    x0, y0, x1, y1 = NOTE_E_BOX
    e_alpha = np.zeros((H, W), np.float32)
    e_alpha[y0:y1, x0:x1] = ared[y0:y1, x0:x1]
    core = e_alpha > 0.9
    ink = np.median(img[core].astype(np.float32), axis=0)
    moved = cv2.warpAffine(e_alpha, np.float32([[1, 0, NOTE_SHIFT[0]], [0, 1, NOTE_SHIFT[1]]]), (W, H), flags=cv2.INTER_LINEAR)
    print("note: 'o' px", int((lab == o_lab).sum()), "ink colour", ink.round().tolist())
    return paint(img, moved, ink)


def run(src=SRC):
    img = np.asarray(Image.open(src).convert("RGB")).copy()
    img = fix_date(img)
    img = fix_lines(img)
    img = fix_note(img)
    return img


def changed_boxes(a, b, thr=3):
    """bounding boxes (x0, y0, x1, y1) of the connected areas where two RGBA pages differ by more than thr"""
    diff = (np.abs(a[..., :3].astype(int) - b[..., :3].astype(int)).max(-1) > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(diff, KERNEL(7)), connectivity=8)
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
