"""mag_sb_01_02_mag2.py [OUT_DIR]: the Global Traveler's Quarterly cover (SB_01_02_MAG2) with its baked lettering put right.
Klein's redraw keeps the title, VOLUME LII / NUMBER TWO and JUNE correctly, but garbles the italic caption ("Hawaiian Paradise's Ancient Mystery
... With 54 Ills."), the year 1937 and the two price strings on the lilac bar ("$3.50 A YEAR", "50¢ THE COPY"). Those are wiped (only their own ink,
the paper filled from the paper round it) and re-set on the straightened 4x picture: the caption in Times Italic, the year in Times Roman, the
prices in Times Bold caps, each stretched to the old print's cap height and span. The ornate frame, the tiki photograph, the title, the labels,
the globe and the rules are not touched.
Writes OUT_DIR/SB_01_02_MAG2_4x_edit.png (straightened) and OUT_DIR/SB_01_02_MAG2_page.png (page frame, via tile_mags_rot.back).
Usage: venv/bin/python -I tools/mag_sb_01_02_mag2.py [OUT_DIR]
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tile_mags_rot as tm

STEM = "SB_01_02_MAG2"
DEFAULT_OUT = Path("/tmp/claude-1000/-home-deck/8d33977a-9b65-41a7-af0e-5ee75413ed5e/scratchpad/mags_text") / STEM
WIN = Path.home() / ".local/share/fonts/winfonts"
TIMES = WIN / "times.ttf"
TIMES_BOLD = WIN / "TIMESBD.TTF"
TIMES_ITALIC = WIN / "Timesi.TTF"

CAPTION_INK = np.array([36, 36, 46], np.float32)          # the caption's dark ink (sampled 31,32,42; softened)
YEAR_INK = np.array([38, 38, 48], np.float32)             # JUNE's ink (sampled 38,38,48)
PRICE_INK = np.array([43, 42, 63], np.float32)            # the price strings on the lilac bar (sampled 43,42,63)


def text_mask(text, font, cap, x0, x1, base, shape):
    """coverage of `text` in the straightened 4x picture: capitals `cap` px high, baseline at row `base`, ink spanning columns x0..x1 (the glyphs
    are stretched to that span, the old print's width)"""
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
    """the paper under a wiped area: a smooth fill from the pixels round it (multi-scale normalised blur) plus grain matched to the paper's own"""
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
    resid = rgb[ring] - out[ring]
    grain = float(1.4826 * np.median(np.abs(resid - np.median(resid)))) * 0.7 if resid.size else 1.5
    noise = np.random.default_rng(seed).normal(0, grain, wipe.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.5)[..., None]
    return np.where(wipe[..., None], np.clip(out + noise, 0, 255), rgb)


def paint(img, a, ink):
    out = img.astype(np.float32)
    return out * (1 - a[..., None]) + np.asarray(ink, np.float32)[None, None, :] * a[..., None]


def rect(shape, x0, x1, y0, y1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


# (text, font, cap, x0, x1, baseline, ink, wipe box (x0, x1, y0, y1), dark threshold of the old ink)
ITEMS = [
    # the caption: Times Italic, cap 21, the old span 327..1005, baseline 1251.5; the dots are the leaders of the old print
    ("Hawaiian Paradise's Ancient Mystery...........With 54 Ills.", TIMES_ITALIC, 21, 327, 1005, 1251.5, CAPTION_INK, (318, 1008, 1226, 1258), 120),
    # the year: Times Roman like JUNE (which is right and stays), cap 19.5, the old span 857..910, baseline 1300
    ("1937", TIMES, 19.5, 857, 910, 1300.0, YEAR_INK, (850, 918, 1276, 1305), 120),
    # the price strings on the lilac bar: Times Bold caps, cap 18, baseline 1352.5
    ("$3.50 A YEAR", TIMES_BOLD, 18, 386, 537, 1352.5, PRICE_INK, (380, 546, 1330, 1358), 110),
    ("50¢ THE COPY", TIMES_BOLD, 18, 773, 940, 1352.5, PRICE_INK, (765, 948, 1330, 1358), 110),
]


def build(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hd = np.asarray(Image.open(ROOT / "tile_job" / f"rot_{STEM}_4x.png").convert("RGB")).astype(np.float32)
    H, W = hd.shape[:2]
    Lum = 0.3 * hd[..., 0] + 0.59 * hd[..., 1] + 0.11 * hd[..., 2]
    wipe = np.zeros((H, W), bool)
    for text, font, cap, x0, x1, base, ink, box, thr in ITEMS:
        m = rect((H, W), *box) & (Lum < thr)                        # only the old ink in each box (the frame and the rules stay)
        wipe |= cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    cur = fill_paper(hd, wipe, seed=4)
    for text, font, cap, x0, x1, base, ink, box, thr in ITEMS:
        a = text_mask(text, font, cap, x0, x1, base, (H, W))
        cur = paint(cur, a, ink)
    edit = np.clip(np.rint(cur), 0, 255).astype(np.uint8)
    ed = out_dir / f"{STEM}_4x_edit.png"
    Image.fromarray(edit).save(ed)
    page = tm.back(STEM, src=ed, dst=out_dir / f"{STEM}_page.png")
    print("wiped", int(wipe.sum()), "px")
    print("edit", ed, "page", page)
    return ed, page


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
