"""mag_sb_11_01_mag4.py [OUT.png]: the sepia 'Hollywood Confidential' cover SB_11_01_MAG4 with its baked-in lettering put right, in the upright 4x frame of tile_mags_rot.
Klein garbles the small print here as well: the HOLLYWOOD band has wrong letters (the whole word is re-set), the logo has an invented x-height glyph after its 'l', the
date is garbled, the 't' of 'Pirate' is malformed (its ascender is missing), the headline has a dim copy under each line, the slanted red caption is garbled with an extra line
below the original's two, and the body column is nonsense.
The wording is the game's own: string 40142 (headline) and 40143 (the zoom article), set line by line as the original prints it: 32 lines in four paragraphs (11 + 8 + 10 + 3),
italics as the string has them. The last lines differ from the string where the original prints them differently ('Lana Cooper, the star of such' / 'films as ...' /
'Wore Red, was first on page 2'). The caption is set at the original's tilt (about 22 degrees). The white curled flap is the redraw's own and is left alone; the date is
clipped at its diagonal left edge, as the original's flap hides the end of the date. The old ink is wiped by a smooth fill of the neighbouring paper and the new lettering is
laid on in a matching face (Bell MT column, Impact caption, Trebuchet bold italic date, DejaVu Sans oblique band). Known limit: where the invented glyph stood the paper
patch is a plain tone with a visible edge (the original has the flap there). Output: tile_mags_rot.back() of the edited picture, so the alpha is the original's.
Usage: venv/bin/python -I tools/mag_sb_11_01_mag4.py [OUT.png]   (default: scratchpad mags_text/SB_11_01_MAG4/page.png)
"""
import re
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tile_mags_rot as tm

STEM = "SB_11_01_MAG4"
SCRATCH = Path("/tmp/claude-1000/-home-deck/8d33977a-9b65-41a7-af0e-5ee75413ed5e/scratchpad/mags_text") / STEM
WIN = Path.home() / ".local/share/fonts/winfonts"
BELL, BELLI = str(WIN / "BELL.TTF"), str(WIN / "BELLI.TTF")
IMPACT = str(WIN / "IMPACT.TTF")
TREB_BI = str(WIN / "Trebucbi.TTF")
DEJAVU_OBL = "/usr/share/fonts/TTF/DejaVuSans-Oblique.ttf"

# the article (game string 40143) as the original prints it: 32 lines in four paragraphs; <I>..<i> marks italics
COLUMN = [
    "The ill-fated Colossal Pictures air epic,", "<I>Baroness of the Air,<i> suffered yet another", "setback yesterday, as air pirate Nathan",
    "Zachary disrupted filming of a critical", "stunt sequence. While Zachary and his", "gang, the Fortune Hunters, attempted",
    "to spirit away glamorous starlet Lana", "Cooper, the pirate leader ruined", "several meticulously planned stunts and",
    "inflicted an estimated $150,000 of", "damage on the sets.",
    "In a Colossal Pictures press conference", "last evening, <I>Baroness<i> director Kurt", "Edward was visibly enraged. \"It wasn't",
    "enough that those hooligans kidnapped", "my leading lady. No, they had to blast", "the place to smithereens. Rockets and",
    "bullets were flying everywhere, planes", "crashing into sets ... it was a disaster.\"",
    "Johnny Johnson, the second-in-command", "at Hughes Aviation (the parent company", "of Colossal Pictures), vowed that Zachary",
    "and his men would not get away with their", "criminal schemes. \"Those vultures have", "picked the wrong target this time,\" he said.",
    "\"Hughes Aviation will not sit idly by and let", "these brigands profit from their crimes. We", "<I>will<i> get Lana Cooper back, and we <I>will<i>",
    "make Zachary pay for what he's done.\"",
    "Lana Cooper, the star of such", "films as <I>The Sands of the Sahara<i> and <I>She",
    "<I>Wore Red,<i> was first on page 2",
]
PARA_START = {0, 11, 19}                                 # the last paragraph (line 29) is not indented in the original
COL_X, COL_INDENT, COL_Y0, COL_PITCH, COL_SIZE = 188, 16, 590, 19.77, 17.6   # left margin, indent, first line centre, pitch, Bell MT px (4x)
COL_BOX = (572, 1212, 176, 528)                                              # the old column (y0, y1, x0, x1); the paper is plain out to x 528 at these rows
BOT_Y = (1212, 1290)                                                         # the redraw's garbled lines below the last one (the original has none there)
EDGE_X0, EDGE_SLOPE = 381.0, 0.0                                            # the paper edge (dark left, beige right) is a straight line at x 380.5 in the original, y 1212-1295
COL_INK = (222, 216, 202)

BAND = "HOLLYWOOD"
BAND_X0, BAND_X1, BAND_BASE, BAND_CAP = 227, 428, 171, 21.0                  # the thin italic band word, ink from x 227 to 428, cap 21 px (4x)
BAND_BOX = (144, 178, 218, 440)
BAND_INK = (235, 225, 210)

LOGO_T_BOXES = [(220, 276, 968, 1032), (276, 306, 962, 1024)]          # the extra x-height glyph after the logo's 'l' (the 'l' stays; the stripe starts at y 313)
DATE, DATE_X, DATE_BASE, DATE_CAP = "May 22, 1937", 935, 357, 17.0           # gold, cap 17 px
DATE_BOX = (332, 366, 928, 1040)                                              # below the stripe (it ends at y 330)
DATE_INK = (232, 178, 58)
FLAP_X0, FLAP_SLOPE = 1026.0, 0.34                                          # the white flap's left edge: x = 1026 + 0.34 * (y - 310)

HEAD_BOX = (340, 575, 200, 560)                                              # the three headline lines (string 40142)
GHOST_DX, GHOST_DY = 1, 15                                                   # the dim copy of the headline sits this far from the letters

ANG = -22.25                                                                 # the caption's tilt (text rises to the right), de-rotated about CAP_C
CAP_C = (700, 1000)
CAPTION = ["Rockets and bullets were flying everywhere,", "planes crashing into sets ... it was a disaster."]
CAP_X, CAP_BASE, CAP_PITCH, CAP_W = 510, 1000, 37, 533                       # de-rotated frame: line 1 spans CAP_W px in the original
CAP_RBOX = (955, 1100, 470, 1060)                                            # de-rotated rows and columns of the old caption
CAP_INK = (186, 38, 36)


def box_of(shape, box):
    m = np.zeros(shape, bool)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = True
    return m


def dilate(m, k=3):
    return cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def wipe(P, mask, seed=3):
    """old ink (mask) replaced by a smooth fill of the unmasked neighbours (normalised convolution at four scales) plus the paper's grain"""
    keep = (~mask).astype(np.float32)
    out = P.copy()
    done = ~mask
    for s in (1.5, 4.0, 12.0, 30.0):
        num = cv2.GaussianBlur(P * keep[..., None], (0, 0), s)
        den = cv2.GaussianBlur(keep, (0, 0), s)
        new = (~done) & (den > 0.02)
        out[new] = (num / np.maximum(den, 1e-6)[..., None])[new]
        done |= new
    grain = cv2.GaussianBlur(np.random.default_rng(seed).normal(0, 1, mask.shape).astype(np.float32), (0, 0), 0.7)
    out[mask] += grain[mask][:, None] * 2.2
    return out


def load():
    return np.asarray(Image.open(tm.ROOT / "tile_job" / f"rot_{STEM}_4x.png").convert("RGB")).astype(np.float32)


def layer(shape, draw):
    """alpha of the text that draw(ImageDraw) writes at 4x, softened like the redraw's own edge"""
    H, W = shape
    im = Image.new("L", (W, H), 0)
    draw(ImageDraw.Draw(im))
    return np.clip(cv2.GaussianBlur(np.asarray(im, np.float32) / 255, (0, 0), 0.5) * 1.15, 0, 1)


def composite(P, a, colour):
    a = a[..., None]
    return P * (1 - a) + np.asarray(colour, np.float32) * a


def band(P):
    """HOLLYWOOD re-set: the old cream letters are wiped (the red rule and the dark paper are refilled), and the word is laid in DejaVu Sans oblique
    with the cap height and the ink width the original has (the letters of the redraw are wrong, so the whole word is set)"""
    R, G, B = P[..., 0], P[..., 1], P[..., 2]
    cream = (np.minimum(np.minimum(R, G), B) > 150) & box_of(R.shape, BAND_BOX)
    P = wipe(P, dilate(cream), seed=21)
    f = ImageFont.truetype(DEJAVU_OBL, 100)
    size = 100 * BAND_CAP / -f.getbbox("H", anchor="ls")[1]
    f = ImageFont.truetype(DEJAVU_OBL, size)
    im = Image.new("L", (int(f.getlength(BAND)) + 80, 160), 0)
    ImageDraw.Draw(im).text((40, 120), BAND, font=f, fill=255, anchor="ls")
    bb = im.getbbox()
    crop = im.crop(bb).resize((BAND_X1 - BAND_X0, bb[3] - bb[1]), Image.LANCZOS)
    a = np.zeros(R.shape, np.float32)
    top = int(round(BAND_BASE - (120 - bb[1])))                                # the baseline lands on BAND_BASE
    a[top:top + crop.height, BAND_X0:BAND_X1] = np.asarray(crop, np.float32) / 255
    a = cv2.GaussianBlur(a, (0, 0), 0.5)
    return composite(P, np.clip(a * 1.15, 0, 1), BAND_INK)


def logo(P):
    """the invented x-height glyph after the logo's 'l' goes (the 'l' stays). Each box (the glyph's top part and its bottom bar) is filled from the grey paper just
    right of it: a per-row mean of a wide strip of that paper, smoothed down the rows so the paper's gradient carries through without stripes. Cream (the logo's
    letters and the flap's white) and the yellow stripe are not used as source; the box edge is feathered into the paper."""
    H, W = P.shape[:2]
    mn = P.min(2)
    cream = dilate(mn > 205, 5)
    yellow = (P[..., 0] - P[..., 2]) > 80
    grey = ((mn > 110) & (mn <= 205) & ~cream & ~yellow).astype(np.float32)
    out = P.copy()
    box = np.zeros((H, W), bool)
    for (y0, y1, x0, x1) in LOGO_T_BOXES:
        box[y0:y1, x0:x1] = True
        xa, xb = x1 + 2, x1 + 40
        num = (P[y0 - 8:y1 + 8, xa:xb] * grey[y0 - 8:y1 + 8, xa:xb, None]).sum(1)
        den = grey[y0 - 8:y1 + 8, xa:xb].sum(1)
        prof = num / np.maximum(den, 1)[:, None]
        prof = cv2.GaussianBlur(prof[:, None, :].astype(np.float32), (0, 0), sigmaX=0.1, sigmaY=5.0)[:, 0, :]
        for y in range(y0, y1):
            out[y, x0:x1] = prof[y - (y0 - 8)]
    grain = cv2.GaussianBlur(np.random.default_rng(23).normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.7)
    out[box] += grain[box][:, None] * 2.0
    w = cv2.GaussianBlur(box.astype(np.float32), (0, 0), 2.5)
    w = np.clip(w * 1.3, 0, 1)[..., None]
    return P * (1 - w) + out * w


def headline_t(P):
    """the 't' of 'Pirate' (line 1) is malformed (its ascender is missing, so it reads as an 'i'). The same face and size gives the right 't' in 'Nathan' (line 2,
    x 270-283, its baseline 70 px lower): its coverage is measured on its own dark paper, moved into the slot (dx +118, dy -70) and laid on the paper of the slot.
    The old glyph is wiped inside x 388-402 only, so the 'a' (ends at x 387) and the 'e' (starts at x 405) are kept."""
    from scipy import ndimage
    H, W = P.shape[:2]
    mn = P.min(2)
    L = P.mean(2)
    cream = mn > 150
    # the source 't' of 'Nathan': its coverage against the dark paper round it, and its core colour
    gy0, gy1, gx0, gx1 = 436, 478, 264, 290
    ring = np.zeros((H, W), bool)
    ring[gy0 - 6:gy1 + 6, gx0 - 6:gx1 + 6] = True
    ring[gy0:gy1, gx0:gx1] = False
    bgL = np.median(L[ring & ~cream])
    gcream = cream[gy0:gy1, gx0:gx1]
    coreL = np.median(L[gy0:gy1, gx0:gx1][gcream])
    alpha_src = np.clip((L - bgL) / max(coreL - bgL, 1.0), 0, 1)
    alpha_src[:, :269] = 0                                                  # only the 't' itself (the 'a' ends at x 267, the 'h' starts at x 284)
    alpha_src[:, 284:] = 0
    core_rgb = np.median(P[gy0:gy1, gx0:gx1][gcream], axis=0)
    # the old 't' of 'Pirate': the cream component that lies inside x 388-402 (the 'a' and the 'e' are other components), plus its halo in that box
    lab, n = ndimage.label(cream[360:420, 380:420])
    old_t = np.zeros((H, W), bool)
    for i, (ys, xs) in enumerate(ndimage.find_objects(lab)):
        if xs.start + 380 >= 388 and xs.stop + 380 <= 403:
            old_t[360 + ys.start:360 + ys.stop, 380 + xs.start:380 + xs.stop] |= lab[ys, xs] == i + 1
    box = np.zeros((H, W), bool)
    box[373:405, 388:403] = True
    wipe_m = box & (cv2.dilate(old_t.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) | box & (mn > 90)     # the old serif foot too
    near = cv2.dilate(wipe_m.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    src = near & ~wipe_m & ~cream
    keep = src.astype(np.float32)
    out = P.copy()
    done = ~wipe_m
    for sg in (1.5, 4.0, 10.0):
        num = cv2.GaussianBlur(P * keep[..., None], (0, 0), sg)
        den = cv2.GaussianBlur(keep, (0, 0), sg)
        new = wipe_m & ~done & (den > 0.02)
        out[new] = (num / np.maximum(den, 1e-6)[..., None])[new]
        done |= new
    out[wipe_m & ~done] = P[src].mean(0)
    # the good 't' in the slot: target (x, y) takes the source coverage at (x - 118, y + 70)
    ty, tx = np.mgrid[373:405, 388:403]
    a = cv2.GaussianBlur(alpha_src[ty + 70, tx - 118].astype(np.float32), (0, 0), 0.35)
    out[373:405, 388:403] = out[373:405, 388:403] * (1 - a[..., None]) + core_rgb[None, None, :] * a[..., None]
    return out


def headline_ghost_fill(P):
    """the dim copy under each headline line is wiped, and the paper is filled from the dark paper round it only (never from the cream letters: their light would
    smear into the copy's place)"""
    mask = headline_ghost(P)
    mn = P.min(2)
    keep = ~(mask | dilate(mn > 150, 4))
    return side_fill(P, mask, keep, seed=31)


def headline_ghost(P):
    """the dim copy under each headline line is the letters again, shifted (1, 15) px at 4x. The copy is dim (under 150 at its brightest); the real letters and their
    one-pixel edge are kept, so the mask is the shifted letters grown by 2 px, minus the dilated letters, and only dim pixels are taken"""
    mn = P.min(2)
    m = box_of(mn.shape, HEAD_BOX)
    letters = (mn > 170) & m
    shifted = cv2.warpAffine(letters.astype(np.float32), np.float32([[1, 0, GHOST_DX], [0, 1, GHOST_DY]]), (mn.shape[1], mn.shape[0]))
    grown = cv2.dilate((shifted > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return grown & ~dilate(letters, 3) & (mn < 150) & m


def date(P):
    """'May 22, 1937' in gold: the redraw's garbled date is wiped (below the yellow stripe, which ends at y 330), and the new date is laid on in Trebuchet bold
    italic, clipped at the white flap's diagonal left edge (x 1026 at y 310, x 1051 at y 388): the flap hides the end of the date, as in the original"""
    R, B = P[..., 0], P[..., 2]
    gold = (R - B > 60) & (R > 110) & box_of(R.shape, DATE_BOX)
    P = wipe(P, dilate(gold), seed=25)
    f = ImageFont.truetype(TREB_BI, 100)
    size = 100 * DATE_CAP / -f.getbbox("H", anchor="ls")[1]
    f = ImageFont.truetype(TREB_BI, size)
    a = layer(R.shape, lambda d: d.text((DATE_X, DATE_BASE), DATE, font=f, fill=255, anchor="ls"))
    ys, xs = np.mgrid[0:R.shape[0], 0:R.shape[1]]
    edge = FLAP_X0 + FLAP_SLOPE * (ys - 310)
    a = a * (xs < edge - 1)
    return composite(P, a, DATE_INK)


def side_fill(P, mask, keep, seed):
    """the wipe of wipe(), but the fill of each masked pixel only takes paper from `keep` (the same side of the paper edge)"""
    out = P.copy()
    kf = keep.astype(np.float32)
    done = ~mask
    for sg in (1.5, 4.0, 12.0, 30.0):
        num = cv2.GaussianBlur(P * kf[..., None], (0, 0), sg)
        den = cv2.GaussianBlur(kf, (0, 0), sg)
        new = (~done) & (den > 0.02)
        out[new] = (num / np.maximum(den, 1e-6)[..., None])[new]
        done |= new
    grain = cv2.GaussianBlur(np.random.default_rng(seed).normal(0, 1, mask.shape).astype(np.float32), (0, 0), 0.7)
    out[mask] += grain[mask][:, None] * 2.2
    return out


def column(P):
    """the garbled column is wiped (every stroke: it deviates from the smooth paper beneath it), then the 32 printed lines are laid on.
    Below the last line the redraw's garbled strokes are cleaned on each side of the paper edge against that side's own paper."""
    y0, y1, x0, x1 = COL_BOX
    g = cv2.cvtColor(P.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg = cv2.medianBlur(g.astype(np.uint8), 31).astype(np.float32)
    H, W = g.shape
    ink = np.zeros(g.shape, bool)
    ink[y0:y1, x0:x1] = np.abs(g - bg)[y0:y1, x0:x1] > 22          # strokes deviate from the smooth paper (beige or photo) either way
    ys = np.arange(H)[:, None]
    X = np.arange(W)[None, :]
    edge = EDGE_X0 + (ys - BOT_Y[0]) * EDGE_SLOPE
    band = (ys >= BOT_Y[0]) & (ys < BOT_Y[1]) & (X >= 184) & (X <= x1)
    dark_side = band & (X < edge - 2)
    beige_side = band & (X > edge + 2)
    wb = (beige_side & (g > 85)).astype(np.float32)                  # the beige paper, smoothly estimated from its own pixels
    Bs = cv2.GaussianBlur(g * wb, (0, 0), 6) / np.maximum(cv2.GaussianBlur(wb, (0, 0), 6), 1e-3)
    strip = (dark_side & (g > 45)) | (beige_side & (np.abs(g - Bs) > 22)) | (band & (np.abs(X - edge) <= 2) & (g > 170))
    sm = dilate(strip)
    P = wipe(P, dilate(ink), seed=27)
    keep = band & ~sm
    P = side_fill(P, sm & (X < edge), keep & (X < edge - 2), seed=27)
    P = side_fill(P, sm & (X >= edge), keep & (X > edge + 2), seed=27)
    # the paper edge itself: a clean straight step at x 380.5 (as in the original), each row blended from the two papers either side of it
    e0 = int(round(EDGE_X0))
    xs = np.arange(e0 - 3, e0 + 4)
    s_ = np.clip((xs - e0 + 1.0) / 2.5, 0, 1)[None, :, None]
    for y in range(BOT_Y[0], BOT_Y[1]):
        Lv = np.median(P[y, e0 - 14:e0 - 5], axis=0)
        Rv = np.median(P[y, e0 + 6:e0 + 15], axis=0)
        P[y, e0 - 3:e0 + 4] = Lv[None, :] * (1 - s_[0]) + Rv[None, :] * s_[0]
    f_r = ImageFont.truetype(BELL, COL_SIZE)
    f_i = ImageFont.truetype(BELLI, COL_SIZE)

    def draw(d):
        for i, line in enumerate(COLUMN):
            x = COL_X + (COL_INDENT if i in PARA_START else 0)
            yb = COL_Y0 + COL_PITCH * i + 4.5
            italic = False
            for seg in re.split(r"(<I>|<i>)", line):
                if seg == "<I>" or seg == "<i>":
                    italic = seg == "<I>"
                    continue
                if seg:
                    f = f_i if italic else f_r
                    d.text((x, yb), seg, font=f, fill=255, anchor="ls")
                    x += f.getlength(seg)
    return composite(P, layer(g.shape, draw), COL_INK)


def caption(P):
    """the red caption, tilted: the old red ink (and its pink fringe) is wiped in the tilted frame, and the two lines are set in Impact at the original's tilt.
    The paper under the old ink is refilled by normalised convolution from the pixels that are not red, so the photo's own shapes come through unchanged."""
    M = cv2.getRotationMatrix2D(CAP_C, ANG, 1.0)                              # upright -> de-rotated
    H, W = P.shape[:2]
    box_d = np.zeros((H, W), np.float32)
    y0, y1, x0, x1 = CAP_RBOX
    box_d[y0:y1, x0:x1] = 1
    box_u = cv2.warpAffine(box_d, M, (W, H), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST) > 0.5
    R, G, B = P[..., 0], P[..., 1], P[..., 2]
    red = ((R - G) > 45) & ((R - B) > 45) & box_u
    wipe_m = cv2.dilate(red.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    wipe_m &= box_u
    P = wipe(P, wipe_m, seed=29)
    f = ImageFont.truetype(IMPACT, 100)
    f = ImageFont.truetype(IMPACT, 100 * CAP_W / f.getlength(CAPTION[0]))

    def draw(d):
        for k, t in enumerate(CAPTION):
            d.text((CAP_X, CAP_BASE + k * CAP_PITCH), t, font=f, fill=255, anchor="ls")
    a_d = layer((H, W), draw)
    a_u = cv2.warpAffine(a_d, M, (W, H), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return composite(P, a_u, CAP_INK)


def run(out=None):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    P = load()
    P = headline_ghost_fill(P)
    P = headline_t(P)
    P = band(P)
    P = logo(P)
    P = date(P)
    P = column(P)
    P = caption(P)
    edited = SCRATCH / "edited_4x.png"
    Image.fromarray(P.clip(0, 255).astype(np.uint8)).save(edited)
    return tm.back(STEM, src=edited, dst=out or SCRATCH / "page.png")


if __name__ == "__main__":
    print(run(sys.argv[1] if len(sys.argv) > 1 else None))
