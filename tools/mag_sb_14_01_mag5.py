"""mag_sb_14_01_mag5.py [SRC_4x.png] [OUT_PAGE.png]: the 'Daily Variety' cover (SB_14_01_MAG5), lettering only.
The tiled redraw garbles three kinds of print: the small mauve 'Daily' (D read as O, i as l), the date tab (AUGUST 1937 read as AUOUST 1837) and the
left column of small white type on the dark studio backdrop (string 40069, the zoom article, set as the print does: 17 lines, Bell MT at the print's size,
ending '(Continued on p. 5)' as the cover shows). The 'Variety' logo, the orange headline and the photograph are not touched.
Each element is wiped with its own ink only (a colour or top-hat mask, dilated, filled by cv2.inpaint from the neighbouring paper or backdrop) and set again at 4x,
in the straightened frame (tile_job/rot_SB_14_01_MAG5_4x.png). Over the blouse the wipe skips tall pieces (the belt ring and strap edges), so the photograph keeps its detail.
The page-frame output is made with tile_mags_rot.back(), which keeps the original's alpha.
  mag_sb_14_01_mag5.py                       the default source -> scratch mags_text/SB_14_01_MAG5/page_out.png (+ page_out_upright.png, the straightened edit)
  mag_sb_14_01_mag5.py SRC.png OUT_PAGE.png  the same from an explicit straightened 4x source, writing OUT_PAGE.png (and OUT_PAGE_upright.png beside it)
"""
import json, re, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tile_mags_rot as tr

STEM = "SB_14_01_MAG5"
STR = {d["id"]: d["text"] for d in json.load(open(Path.home() / "Games/CSVM/extracted/rof/ui_strings.json"))}
FD = Path.home() / ".local/share/fonts/winfonts"
BELL, ARIAL_B, TAHOMA = str(FD / "BELL.TTF"), str(FD / "arialbd.ttf"), str(FD / "tahoma.ttf")

# column: Bell MT (the game's BEL face) at the print's size, lines on the print's baselines and pitch (1x -> 4x)
COL_SIZE = 21.6                                                    # 5.4 px at 1x: the first line of the print measures 352 px at 4x, Bell gives 354
COL_X, COL_INDENT, COL_RIGHT = 136, 28, 576                        # 34 px at 1x (paragraph indent 7 px), ragged measure to 144 px at 1x
COL_BASE, COL_PITCH = 760.0, 23.25                                 # baseline of the first line and the leading (5.81 px at 1x)
COL_INK = (216, 216, 214)
COL_STROKE, COL_BLUR, COL_GAIN = 1, 1.3, 1.0                     # the print is a heavier, softer blur than the redraw's hairline type: a 1 px stroke at 4x, then a soft blur
CONTINUED = "(Continued on p. 5)"                                  # the cover's own closing line (the string says '(More on "Cooper," p. 5.)')
COL_BOX = (118, 728, 584, 1168)                                    # x0, y0, x1, y1 of the old column ink (4x)
COL_T_STRONG, COL_T_FAINT = 60, 20                                 # top-hat thresholds of the old type: strong ink anywhere, faint ink too (see COL_DARK_X)
COL_STRONG_H = 26                                                  # the tallest bright piece that is a letter or a word (the ring and strap edges are taller)
COL_GHOST_H, COL_GHOST_W = 18, 120                                 # over the blouse, a faint piece no taller than a line and no wider than a word (the belt and blouse edges are longer)
COL_REPAIR = [(516, 852, 544, 874), (476, 882, 500, 904)]        # 4x boxes of stray old type on the blouse edge (the 'in' on the ring, a faint line-7 ghost)
COL_DARK_X = 470                                                   # x (4x) where the blouse starts to the right of the column's dark backdrop

# 'Daily': a small sans in the print's mauve; ink box 243..324 x, cap height 30 px, baseline 188.5 (4x)
DAILY_BOX = (236, 150, 340, 200)
DAILY = dict(text="Daily", x=243.0, base=188.5, cap=30.0, width=81.0, ink=(192, 112, 116), font=TAHOMA)

# date tab: a bold sans in near-black; cap height 17 px and width 156 px at x 908 (4x), baseline 125.6
DATE_BOX = (900, 100, 1086, 134)                                   # the black bar under the tab starts at y 136
DATE = dict(text="AUGUST 1937", x=908.0, base=125.6, cap=17.0, width=156.0, ink=(24, 24, 28), font=ARIAL_B)


def wrap(text, font, width, indent):
    """greedy wrap at `width` px; the first line of a paragraph starts `indent` px in; a hyphen is a break point (the print breaks 'actor-friendly')"""
    toks = [t for t in re.split(r"(?<=-)|\s+", text) if t]
    out, cur = [], ""
    for w in toks:
        sep = "" if (not cur or cur.endswith("-")) else " "
        t = cur + sep + w if cur else w
        if font.getlength(t) <= width - (indent if not out else 0) or not cur:
            cur = t
        else:
            out.append(cur); cur = w
    out.append(cur)
    return out


def column_paragraphs():
    """string 40069 with its own paragraphs (each starts with five spaces), its closing line replaced by the cover's own"""
    body = re.sub(r"^\[[A-Z0-9]+\]", "", STR[40069], flags=re.M)
    paras = [re.sub(r"\s+", " ", p).strip() for p in body.split("\n") if p.strip()]
    return paras[:-1] + [CONTINUED]


def column_lines(paras):
    """(text, x, baseline) for the 17 lines: the first line of each paragraph is indented, the closing line is not"""
    f = ImageFont.truetype(BELL, COL_SIZE)
    items, n = [], 0
    for pi, p in enumerate(paras):
        indent = COL_INDENT if pi < len(paras) - 1 else 0
        for li, ln in enumerate(wrap(p, f, COL_RIGHT - COL_X, indent)):
            x = COL_X + (indent if li == 0 else 0)
            items.append((ln, x, COL_BASE + COL_PITCH * n)); n += 1
    return items


def paste_text(H, W, text, font_path, size, x, base, cap, width):
    """one word at 4x: the glyphs drawn at `size`, then stretched so its cap height is `cap` px and its ink is `width` px wide, its left edge at x and its baseline at base"""
    f = ImageFont.truetype(font_path, size)
    l, t, r, b = f.getbbox(text, anchor="ls")                      # ink box relative to the origin (left, baseline)
    ct = f.getbbox("H", anchor="ls")[1]                            # top of a capital H: the cap height of this face
    pad = 8
    im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - l, pad - t), text, font=f, fill=255, anchor="ls")
    ox, oy = pad - l, pad - t                                      # the origin inside the small picture
    sx, sy = width / (r - l), cap / float(-ct)
    im = im.resize((max(1, round(im.width * sx)), max(1, round(im.height * sy))), Image.LANCZOS)
    layer = np.zeros((H, W), np.float32)
    px, py = int(round(x - ox * sx)), int(round(base - oy * sy))
    ax, ay = max(0, px), max(0, py)
    a = np.asarray(im, np.float32) / 255
    a = a[ay - py:, ax - px:]
    h, w = min(a.shape[0], H - ay), min(a.shape[1], W - ax)
    layer[ay:ay + h, ax:ax + w] = a[:h, :w]
    return layer


def paint(rgb, alpha, ink):
    """lay the ink over the picture by the layer's alpha"""
    out = rgb.astype(np.float32) * (1 - alpha[..., None]) + np.asarray(ink, np.float32)[None, None, :] * alpha[..., None]
    return out.clip(0, 255).astype(np.uint8)


def wipe(rgb, mask, radius=5):
    """the old ink's pixels filled from the neighbouring paper or backdrop (cv2.inpaint), no blur patch"""
    return cv2.inpaint(rgb, mask.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)


def run(src, dst):
    rgb = np.asarray(Image.open(src).convert("RGB")).copy()
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 1) 'Daily': the mauve letters only (the orange shadow of the logo and the white letters are not mauve)
    x0, y0, x1, y1 = DAILY_BOX
    sub = rgb[y0:y1, x0:x1].astype(np.int32)
    mauve = (sub[..., 0] - sub[..., 1] > 22) & (sub[..., 2] > 35) & (sub[..., 0] > 80) & (sub[..., 0] < 250)
    wm = np.zeros((H, W), bool); wm[y0:y1, x0:x1] = mauve
    wm = cv2.dilate(wm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    rgb = wipe(rgb, wm, 4)
    a = paste_text(H, W, DAILY["text"], DAILY["font"], 42, DAILY["x"], DAILY["base"], DAILY["cap"], DAILY["width"])
    rgb = paint(rgb, np.clip(cv2.GaussianBlur(a, (0, 0), 0.9) * 1.1, 0, 1), DAILY["ink"])

    # 2) date tab: the dark letters of AUGUST 1937 (the black bar below them is outside the box)
    x0, y0, x1, y1 = DATE_BOX
    dark = gray[y0:y1, x0:x1] < 110
    wd = np.zeros((H, W), bool); wd[y0:y1, x0:x1] = dark
    wd = cv2.dilate(wd.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    rgb = wipe(rgb, wd, 4)
    a = paste_text(H, W, DATE["text"], DATE["font"], 24, DATE["x"], DATE["base"], DATE["cap"], DATE["width"])
    rgb = paint(rgb, np.clip(cv2.GaussianBlur(a, (0, 0), 0.8) * 1.05, 0, 1), DATE["ink"])

    # 3) the column: the old white type and its faint ghost wiped with their own ink (top-hat), then the 17 lines of the string
    x0, y0, x1, y1 = COL_BOX
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    strong = np.zeros((H, W), np.uint8)
    strong[y0:y1, x0:x1] = th[y0:y1, x0:x1] > COL_T_STRONG              # bright type: strong contrast
    n, lab, st, _ = cv2.connectedComponentsWithStats(strong, connectivity=8)
    ok = np.ones(n, bool); ok[0] = False
    ok[1:] = st[1:, cv2.CC_STAT_HEIGHT] <= COL_STRONG_H                 # over the blouse a tall piece is the belt ring or a strap edge, not a line of type
    wc = (ok[lab] & (strong > 0))
    wc[:, :COL_DARK_X] = strong[:, :COL_DARK_X] > 0                     # over the dark backdrop nothing else is bright
    faint = np.zeros((H, W), np.uint8)
    faint[y0:y1, x0:x1] = th[y0:y1, x0:x1] > COL_T_FAINT                # the faint ghost of the old lines
    wc[y0:y1, x0:COL_DARK_X] |= faint[y0:y1, x0:COL_DARK_X] > 0         # over the dark backdrop every faint piece is the old type
    n, lab, st, _ = cv2.connectedComponentsWithStats(faint, connectivity=8)  # over the blouse only small, letter-sized pieces (its texture is faint too)
    keep = np.zeros(n, bool); keep[1:] = (st[1:, cv2.CC_STAT_HEIGHT] <= COL_GHOST_H) & (st[1:, cv2.CC_STAT_WIDTH] <= COL_GHOST_W)
    wc[:, COL_DARK_X:] |= keep[lab][:, COL_DARK_X:]
    for bx0, by0, bx1, by1 in COL_REPAIR:                               # two stray pieces of the old type fused to the ring and the strap: wiped by hand
        wc[by0:by1, bx0:bx1] |= faint[by0:by1, bx0:bx1] > 0
    wc = cv2.dilate(wc.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    rgb = wipe(rgb, wc, 3)
    items = column_lines(column_paragraphs())
    assert len(items) == 17, len(items)
    f = ImageFont.truetype(BELL, COL_SIZE)
    lay = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(lay)
    for txt, x, yb in items:
        dr.text((x, yb), txt, font=f, fill=255, anchor="ls", stroke_width=COL_STROKE, stroke_fill=255)
    alpha = cv2.GaussianBlur(np.asarray(lay, np.float32) / 255, (0, 0), COL_BLUR)
    rgb = paint(rgb, np.clip(alpha * COL_GAIN, 0, 1), COL_INK)
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
