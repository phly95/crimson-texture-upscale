"""mag_sb_11_01_mag3.py [OUT.png]: the 'Hollywood Confidential' cover SB_11_01_MAG3 with its baked-in lettering put right, in the upright 4x frame of tile_mags_rot.
Klein garbles the small print: HOLLYWOOD ends in an O, the date is unreadable, the red caption is garbled (its lines are set lower than the original's, with a dim copy of a line above them), the
headline has a dim yellow copy under each line, and the body column is nonsense. The wording is the game's own: string 40140 (headline) and 40141 (the zoom article),
set here line by line as the original prints it: 28 lines in three paragraphs, italics as the string has them, ending at 'taking Baroness to'.
Each element: the old ink is wiped (a smooth fill of the neighbouring paper, grain kept) and the new lettering is laid on in a matching face (Bell MT for the column,
Impact for the caption, Trebuchet bold italic for the date, DejaVu Condensed bold oblique for the D). The headline letters, the logo, the 13 cents badge, the photo,
the portrait, the stars and the skull are untouched. Output: tile_mags_rot.back() of the edited picture, so the alpha is the original's.
Usage: venv/bin/python -I tools/mag_sb_11_01_mag3.py [OUT.png]   (default: scratchpad mags_text/SB_11_01_MAG3/page.png)
"""
import re
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tile_mags_rot as tm

STEM = "SB_11_01_MAG3"
SCRATCH = Path("/tmp/claude-1000/-home-deck/8d33977a-9b65-41a7-af0e-5ee75413ed5e/scratchpad/mags_text") / STEM
WIN = Path.home() / ".local/share/fonts/winfonts"
BELL, BELLI = str(WIN / "BELL.TTF"), str(WIN / "BELLI.TTF")
IMPACT = str(WIN / "IMPACT.TTF")
TREB_BI = str(WIN / "Trebucbi.TTF")
DEJAVU_CB = "/usr/share/fonts/TTF/DejaVuSansCondensed-BoldOblique.ttf"

# the article (game string 40141) as the original prints it: 28 lines; <I>..<i> marks italics; PARA_START lines are indented
COLUMN = [
    "The troubled production of <I>Baroness of the<i>", "<I>Air<i> received an unexpected shot in the arm",
    "during a recent pirate raid on the Colossal", "Studios film lot. Air pirate Nathan Zachary",
    "(who had infiltrated a squadron of stunt", "flyers during the filming of a critical chase",
    "sequence) put on such a spectacular", "display of stunt flying that several minutes",
    "of footage that features the inadvertent", "screen personality will appear in the final", "cut of the movie.",
    "\"Well, that pirate was just", "amazing! I mean, I'm constantly screaming",
    "at my stunt flyers to pull off maneuvers half as", "exciting. He dodged! He swooped! It was",
    "spectacular ... simply first-rate flying. It's like", "I always say: We should fire the union flyers",
    "and hire the pirates,\" said Kurt Edwards,", "director of <I>Baroness.<i> \"There were explosions,",
    "bullets zipping past my head, tremendous", "damage to the sets ... it was a total mess.",
    "I must admit, though, the footage we got was", "spectacular. Absolutely amazing. I've seen",
    "some fancy flying in my day, but this was", "breathtaking!\"",
    "Edwards, who many insiders", "believed was on the verge of losing his job,",
    "seems to be back on track for taking <I>Baroness<i> to",
]
PARA_START = {0, 11, 25}
COL_X, COL_INDENT, COL_Y0, COL_PITCH, COL_SIZE = 156, 16, 527.5, 19.52, 17.5   # left margin, indent, first line centre, pitch, Bell MT px (4x)
COL_BOX = (512, 1105, 132, 495)                                              # the old column (y0, y1, x0, x1)
COL_INK = (228, 222, 230)

CAPTION = ["He dodged! He swooped! It was", "spectacular ..."]                # string 40141, the red quote of the article
CAP_X, CAP_BASE, CAP_PITCH, CAP_W = 360, 1196, 46, 482                       # line 1 spans CAP_W px in the original
CAP_BOX = (1150, 1320, 340, 870)
CAP_INK = (178, 18, 24)

DATE, DATE_X, DATE_BASE, DATE_W = "May 22, 1937", 910, 331, 130               # gold, 130 px wide in the original
DATE_BOX = (304, 340, 900, 1052)
DATE_INK = (232, 178, 58)

HEAD_BOX = (340, 650, 230, 1010)                                             # the three headline lines (string 40140)
GHOST_DX, GHOST_DY = -2, 20                                                  # the redraw's dim copy of the headline sits this far from the letters
DW_SLOT = (127, 157, 382, 412)                                               # the O at the end of HOLLYWOOD that must read D
BAND_BASE, BAND_CAP, BAND_INK = 153.5, 23.0, (232, 226, 214)


def load():
    """the straightened 4x picture of the cover (rot_<STEM>_4x.png) as float RGB"""
    return np.asarray(Image.open(tm.ROOT / "tile_job" / f"rot_{STEM}_4x.png").convert("RGB")).astype(np.float32)


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


def paint(P, draw, colour):
    """lay the text that draw(ImageDraw) writes on a 4x layer in colour, softened to the redraw's own edge"""
    H, W = P.shape[:2]
    im = Image.new("L", (W, H), 0)
    draw(ImageDraw.Draw(im))
    a = cv2.GaussianBlur(np.asarray(im, np.float32) / 255, (0, 0), 0.5)
    a = np.clip(a * 1.15, 0, 1)[..., None]
    return P * (1 - a) + np.asarray(colour, np.float32) * a


def side_fill(P, mask, keep, seed):
    """the wipe of wipe(), but the fill of each masked pixel only takes paper from `keep` (never from the letters beside the mask)"""
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


def headline_ghost(P):
    """the dim copy under each headline line is the letters again, shifted (-2, +20) px at 4x: the shifted letters grown by 3 px (the copy is blurred), minus the letters grown by 3"""
    R, B = P[..., 0], P[..., 2]
    m = box_of(R.shape, HEAD_BOX)
    letters = (R - B > 110) & (R > 150) & m
    shifted = cv2.warpAffine(letters.astype(np.float32), np.float32([[1, 0, GHOST_DX], [0, 1, GHOST_DY]]), (R.shape[1], R.shape[0]))
    grown = cv2.dilate((shifted > 0.5).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return grown & ~dilate(letters, 3) & m


def headline_ghost_fill(P):
    """the dim copy is wiped and its paper filled from the paper of its own side: purple paper from purple paper, black from black (a fill across the purple/black
    edge would smear dark into the purple). Never from the gold letters, which would smear into the copy's place"""
    R, G, B = P[..., 0], P[..., 1], P[..., 2]
    mask = headline_ghost(P)
    letters = (R - B > 110) & (R > 150)
    keep = ~(mask | dilate(letters, 4))
    purple = (B - G) > 35
    kp, kb = keep & purple, keep & ~purple
    fill_p = side_fill(P, mask, kp, seed=13)
    fill_b = side_fill(P, mask, kb, seed=13)
    dens = cv2.GaussianBlur(kp.astype(np.float32), (0, 0), 6) / np.maximum(cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 6), 1e-3)
    out = P.copy()
    is_p = (dens > 0.5) & mask
    out[is_p] = fill_p[is_p]
    out[mask & ~is_p] = fill_b[mask & ~is_p]
    return out


def repair_band_D(P):
    """the O at the end of HOLLYWOOD becomes a D: its cream ink is wiped and a bold oblique D of the band's cap height is laid in the slot"""
    R, G, B = P[..., 0], P[..., 1], P[..., 2]
    cream = (np.minimum(np.minimum(R, G), B) > 150) & box_of(R.shape, DW_SLOT)
    P = wipe(P, dilate(cream), seed=5)
    f = ImageFont.truetype(DEJAVU_CB, 32)
    cap = -f.getbbox("H", anchor="ls")[1]
    f = ImageFont.truetype(DEJAVU_CB, 32 * BAND_CAP / cap)
    bb = f.getbbox("D", anchor="ls")
    x_left = DW_SLOT[2] + 2                                                  # the D's stem starts just inside the slot
    return paint(P, lambda d: d.text((x_left - bb[0], BAND_BASE), "D", font=f, fill=255, anchor="ls"), BAND_INK)


def paper_background(g, purple, stroke_guess):
    """the smooth paper under the column, per paper class: purple paper (B-G >= 45) and black paper, normalised convolution over the paper pixels only (the
    strokes are left out), blended by the local purple density, so the purple/black edge does not put a false background along itself"""
    pur_paper = purple & ~stroke_guess
    blk_paper = ~purple & ~stroke_guess
    def nc(v, m, sg):
        return cv2.GaussianBlur(v * m, (0, 0), sg) / np.maximum(cv2.GaussianBlur(m.astype(np.float32), (0, 0), sg), 1e-3)
    bg_p = nc(g, pur_paper.astype(np.float32), 12)
    bg_b = nc(g, blk_paper.astype(np.float32), 12)
    pm = nc(purple.astype(np.float32), (pur_paper | blk_paper).astype(np.float32), 12)
    return pm * bg_p + (1 - pm) * bg_b


def column(P):
    """the garbled column is wiped: every stroke is a local deviation from the paper under it (the purple or the black paper, whichever is there; the dim ghosts
    on the black are caught too). The paper is never masked, so the fill has paper to take its colour from. Then the 28 printed lines are laid on."""
    y0, y1, x0, x1 = COL_BOX
    g = cv2.cvtColor(P.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    purple = (P[..., 2] - P[..., 1]) >= 45
    guess = (g > 120) | ((g > 40) & ~purple)                        # where strokes are likely: left out of the paper estimate
    bg = paper_background(g, purple, guess)
    ink = np.zeros(g.shape, bool)
    ink[y0:y1, x0:x1] = (np.abs(g - bg) > 22)[y0:y1, x0:x1]
    P = wipe(P, dilate(ink), seed=7)
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
    return paint(P, draw, COL_INK)


def caption(P):
    """the red caption: its wrong ink and the dim copy of its first line are wiped, and the two lines are set in Impact where the original has them"""
    R, G, B = P[..., 0], P[..., 1], P[..., 2]
    red = (R - np.maximum(G, B) > 22) & (R > 40) & box_of(R.shape, CAP_BOX)
    P = wipe(P, dilate(red), seed=9)
    f = ImageFont.truetype(IMPACT, 34)
    f = ImageFont.truetype(IMPACT, 34 * CAP_W / f.getlength(CAPTION[0]))

    def draw(d):
        for k, t in enumerate(CAPTION):
            d.text((CAP_X, CAP_BASE + k * CAP_PITCH), t, font=f, fill=255, anchor="ls")
    return paint(P, draw, CAP_INK)


def date(P):
    """'May 22, 1937' in gold, the old date wiped from the purple"""
    R, B = P[..., 0], P[..., 2]
    gold = (R - B > 60) & (R > 110) & box_of(R.shape, DATE_BOX)
    P = wipe(P, dilate(gold), seed=11)
    f = ImageFont.truetype(TREB_BI, 20)
    f = ImageFont.truetype(TREB_BI, 20 * DATE_W / f.getlength(DATE))
    return paint(P, lambda d: d.text((DATE_X, DATE_BASE), DATE, font=f, fill=255, anchor="ls"), DATE_INK)


def run(out=None):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    P = load()
    P = headline_ghost_fill(P)
    P = repair_band_D(P)
    P = date(P)
    P = column(P)
    P = caption(P)
    edited = SCRATCH / "edited_4x.png"
    Image.fromarray(P.clip(0, 255).astype(np.uint8)).save(edited)
    return tm.back(STEM, src=edited, dst=out or SCRATCH / "page.png")


if __name__ == "__main__":
    print(run(sys.argv[1] if len(sys.argv) > 1 else None))
