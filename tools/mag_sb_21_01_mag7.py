"""mag_sb_21_01_mag7.py [SRC_4x.png] [OUT_PAGE.png]: the 'New York Weekly' cover (SB_21_01_MAG7), lettering only.
The tiled redraw garbles four kinds of print: the slogan under the logo (THE NEWS YOU CAN USE. read as THE AIETUS YOU CAN RISE.), the white caption over the skyline
(POLICE HARASS CITIZENS!), the red date band down the left black strip (November 11, 1937, rotated 90 degrees) and the body column at the foot of the page.
The logo, the '10 cents' price, the target emblem, the photograph and the paper are not touched.
Each element is wiped with its own ink only (a darkness, brightness or colour mask against a local paper estimate, dilated, filled by cv2.inpaint) and set again at 4x in
the straightened frame (tile_job/rot_SB_21_01_MAG7_4x.png). The body is the Kurt Edwards quote as printed on this cover (read from the original, where it differs from
string 40141, the MAG3 article's draft); the words the print leaves unreadable come from 40141 in the same voice. The page-frame output is made with tile_mags_rot.back().
  mag_sb_21_01_mag7.py                       the default source -> scratch mags_text/SB_21_01_MAG7/page_out.png (+ page_out_upright.png, the straightened edit)
  mag_sb_21_01_mag7.py SRC.png OUT_PAGE.png  the same from an explicit straightened 4x source, writing OUT_PAGE.png (and OUT_PAGE_upright.png beside it)
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tile_mags_rot as tr

STEM = "SB_21_01_MAG7"
FD = Path.home() / ".local/share/fonts/winfonts"
BELL, ARIAL_B = str(FD / "BELL.TTF"), str(FD / "arialbd.ttf")
IMPACT = str(FD / "IMPACT.TTF")
DEJA_C_B = "/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf"

# slogan: a heavy condensed italic under the logo; 'THE NEWS' is printed in a lighter grey than 'YOU CAN USE.'. The redraw's own band (ink 493..807, caps 330..348 at 4x)
# is kept, since its red stripe sits just below it; the print's band is about 3 px lower and the stripe with it
SLOGAN = dict(text="THE NEWS YOU CAN USE.", split=8, x0=494.0, x1=806.0, base=348.5, cap=18.5, font=IMPACT, shear=0.18,
              ink_a=(110, 106, 102), ink_b=(40, 36, 34), box=(486, 324, 814, 352))
# caption: white capitals over the skyline (4x: 345..700 wide, the width the redraw drew its ghost caption at, cap top 804, baseline 826)
CAPTION = dict(text="POLICE HARASS CITIZENS!", x0=345.0, x1=700.0, base=826.0, cap=21.0, font=ARIAL_B,
               ink=(226, 223, 218), box=(318, 780, 725, 845))
# the red date down the left black strip, reading top to bottom (4x: x 260..278, y 960..1348)
DATE = dict(text="November 11, 1937", x=260.0, y0=960.0, length=388.0, cap=18.0, font=DEJA_C_B, ink=(150, 18, 18), box=(248, 950, 292, 1432))

# body: the Kurt Edwards quote as printed (17 lines: 14 of the quote, 3 of the Edwards paragraph), Bell MT at the print's measure.
# Readings at 1x that are not certain: 'Yeah' (or 'Well'), 'pull off' (or 'put off'; string 40141 has 'pull'), and the closing '(see page 2).'.
BODY_LINES = [
    ("“Yeah, that pirate was just", 0),
    ("amazing,\" I mean, I'm constantly screaming", 0),
    ("at my stunt flyers to pull off maneuvers half as", 0),
    ("exciting. He dodged! He swooped! It was", 0),
    ("spectacular ... simply first-rate flying. It's like", 0),
    ("I always say: We should fire the union flyers", 0),
    ("and hire the pirates,\" said Kurt Edwards,", 0),
    ("director of Baroness. \"There were explosions,", 0),
    ("bullets zipping past my head, tremendous", 0),
    ("damage to the sets ... it was a total mess.", 0),
    ("I must admit, though, the footage we got was", 0),
    ("spectacular. Absolutely amazing. I've seen", 0),
    ("some fancy flying in my day, but this was", 0),
    ("breathtaking!\"", 0),
    ("Edwards, who many insiders", 1),
    ("believed was on the verge of losing his job,", 0),
    ("seems to be back on track (see page 2).", 0),
]
BODY_X, BODY_INDENT = 340.0, 20.0                                  # 85 px at 1x (the paragraph's second block is indented 5 px)
BODY_BASE, BODY_PITCH = 1014.0, 20.24                              # first baseline and leading (253.5 px and 5.06 px at 1x)
BODY_SIZE = 17.5                                                   # Bell MT: the longest line (3) measures about 338 px at 4x, as the print does
BODY_INK = (66, 60, 54)                                            # the print is paler and softer than a crisp face: a light ink and a blur
BODY_BOX = (320, 984, 690, 1412)                                   # the redraw's old column runs about 4 lines past the print's last one (to y 1408)


def ink_image(text, font_path, size, L, C, split=None, shear=0.0):
    """the glyphs of `text` at `size`, its ink box stretched to L x C px (sheared by `shear` for an italic); returns a list of float masks: one, or two when `split` is given
    (the second is text[split:]), each on the same canvas"""
    f = ImageFont.truetype(font_path, size)
    l, t, r, b = f.getbbox(text, anchor="ls")
    pad = 8
    W0, H0 = r - l + 2 * pad, b - t + 2 * pad
    parts = []
    for k in range(2 if split else 1):
        im = Image.new("L", (W0, H0), 0)
        d = ImageDraw.Draw(im)
        if k == 0:
            d.text((pad - l, pad - t), text[:split] if split else text, font=f, fill=255, anchor="ls")
        else:
            d.text((pad - l + f.getlength(text[:split]), pad - t), text[split:], font=f, fill=255, anchor="ls")
        box = im.crop((pad, pad, pad + (r - l), pad + (b - t)))
        if shear:                                                  # an oblique: the top of each row moves right by shear x its height above the baseline
            hh = box.height; extra = int(np.ceil(shear * hh))
            box = box.transform((box.width + extra, hh), Image.AFFINE, (1, shear, -shear * hh, 0, 1, 0), resample=Image.BICUBIC)
        ink = np.asarray(box.resize((int(round(L)), int(round(C))), Image.LANCZOS), np.float32) / 255
        parts.append(ink)
    return parts


def put(layer, img, x, y):
    """paste a float mask into the layer at (x, y), clipped to the layer"""
    H, W = layer.shape
    xi, yi = int(round(x)), int(round(y))
    h, w = img.shape
    ax, ay = max(0, xi), max(0, yi)
    bx, by = min(W, xi + w), min(H, yi + h)
    if bx > ax and by > ay:
        layer[ay:by, ax:bx] = np.maximum(layer[ay:by, ax:bx], img[ay - yi:by - yi, ax - xi:bx - xi])


def paint(rgb, alpha, ink):
    out = rgb.astype(np.float32) * (1 - alpha[..., None]) + np.asarray(ink, np.float32)[None, None, :] * alpha[..., None]
    return out.clip(0, 255).astype(np.uint8)


def wipe(rgb, mask, radius=3):
    return cv2.inpaint(rgb, mask.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)


def box_mask(H, W, box, m):
    """the mask m (H x W bool) limited to the box (x0, y0, x1, y1)"""
    x0, y0, x1, y1 = box
    out = np.zeros((H, W), bool); out[y0:y1, x0:x1] = m[y0:y1, x0:x1]
    return out


def wrap_lines():
    """body lines as printed; a line flagged 1 starts a paragraph (indented)"""
    items = []
    for i, (t, ind) in enumerate(BODY_LINES):
        items.append((t, BODY_X + (BODY_INDENT if ind else 0), BODY_BASE + BODY_PITCH * i))
    return items


def run(src, dst):
    rgb = np.asarray(Image.open(src).convert("RGB")).copy()
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    f32 = gray.astype(np.float32)

    # 1) slogan: the dark letters against the white strip under the logo (a light paper: anything well darker than the paper is ink)
    paper = cv2.dilate(gray, np.ones((9, 9), np.uint8)).astype(np.float32)
    m = box_mask(H, W, SLOGAN["box"], (f32 < paper - 45))
    rgb = wipe(rgb, cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0, 3)
    a, b = ink_image(SLOGAN["text"], SLOGAN["font"], 40, SLOGAN["x1"] - SLOGAN["x0"], SLOGAN["cap"], split=SLOGAN["split"], shear=SLOGAN["shear"])
    la, lb = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    top = SLOGAN["base"] - SLOGAN["cap"]
    put(la, a, SLOGAN["x0"], top); put(lb, b, SLOGAN["x0"], top)
    rgb = paint(rgb, cv2.GaussianBlur(la, (0, 0), 0.6) * 1.05, SLOGAN["ink_a"])
    rgb = paint(rgb, cv2.GaussianBlur(lb, (0, 0), 0.6) * 1.05, SLOGAN["ink_b"])

    # 2) caption: the white capitals over the skyline (the bright letters only: a building's bright edge is not ink)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = box_mask(H, W, CAPTION["box"], (th > 40) & (gray > 175))
    rgb = wipe(rgb, cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0, 3)
    c = ink_image(CAPTION["text"], CAPTION["font"], 40, CAPTION["x1"] - CAPTION["x0"], CAPTION["cap"])[0]
    lc = np.zeros((H, W), np.float32)
    put(lc, c, CAPTION["x0"], CAPTION["base"] - CAPTION["cap"])
    alpha = cv2.GaussianBlur(lc, (0, 0), 0.9)
    rgb = paint(rgb, np.clip(alpha * 1.0, 0, 1), CAPTION["ink"])

    # 3) the red date: red glyphs only (the red strip on the cover's edge is left of the box)
    red = (rgb[..., 0].astype(np.int32) - rgb[..., 1].astype(np.int32)) > 35
    m = box_mask(H, W, DATE["box"], red)
    rgb = wipe(rgb, cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0, 3)
    d = ink_image(DATE["text"], DATE["font"], 40, DATE["length"], DATE["cap"])[0]
    d = np.ascontiguousarray(np.rot90(d, k=-1))                    # clockwise: reads top to bottom, the glyph tops face right
    ld = np.zeros((H, W), np.float32)
    put(ld, d, DATE["x"], DATE["y0"])
    rgb = paint(rgb, cv2.GaussianBlur(ld, (0, 0), 0.6) * 1.1, DATE["ink"])

    # 4) the body column: the dark type against the sepia paper, wiped with its own ink, then the 17 lines
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    paper = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))).astype(np.float32)
    m = box_mask(H, W, BODY_BOX, gray.astype(np.float32) < paper - 38)
    rgb = wipe(rgb, cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0, 3)
    lay = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(lay)
    f = ImageFont.truetype(BELL, BODY_SIZE)
    items = wrap_lines()
    for txt, x, yb in items:
        dr.text((x, yb), txt, font=f, fill=255, anchor="ls")
    alpha = cv2.GaussianBlur(np.asarray(lay, np.float32) / 255, (0, 0), 1.2)
    rgb = paint(rgb, np.clip(alpha, 0, 1), BODY_INK)
    Image.fromarray(rgb).save(dst)
    return items


if __name__ == "__main__":
    OUT = Path("/tmp/claude-1000/-home-deck/8d33977a-9b65-41a7-af0e-5ee75413ed5e/scratchpad/mags_text") / STEM
    src = sys.argv[1] if len(sys.argv) > 1 else ROOT / "tile_job" / f"rot_{STEM}_4x.png"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / "page_out.png"
    upright = dst.with_name(dst.stem + "_upright.png")
    items = run(src, upright)
    for t, x, yb in items:
        print(round(yb, 1), x, t)
    print("upright", upright)
    print("page", tr.back(STEM, src=upright, dst=dst))
