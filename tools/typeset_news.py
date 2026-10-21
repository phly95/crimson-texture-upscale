"""typeset_news.py IN.png OUT.png: SB_02_01_NEWS4 (the Aloha Daily scrap). The small column text of the thumbnail is a blur of a different draft from the game's article
(string 40025). The column is re-set from the game string in Times New Roman (TNR14) at the old print's line pitch, a first-line indent per paragraph, ragged right, running off
the torn bottom like the original; the old ink is wiped inside the column only, so the masthead, the headline and the photograph are untouched."""
import re, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import typeset_page as tp

TIMES = str(tp.F / "times.ttf")
X0, X1, Y0, Y1 = 42, 304, 349, 760
FIRST, PITCH, SIZE, INDENT = 376.5, 17.8, 13.6, 18


def run(src, dst):
    im = np.asarray(Image.open(src).convert("RGBA"))
    pg = tp.Page(im, 0.6, (0.48, 0.9), angle=-2.0)
    H, W = pg.H, pg.W
    gray = cv2.cvtColor(pg.rect[..., :3], cv2.COLOR_RGB2GRAY)
    arect = cv2.warpAffine(im[..., 3], pg.M, (W, H))
    def left(yc):                                                       # the torn left edge leans: lines start 10 px inside it
        cols = np.nonzero(arect[int(yc) - 6:int(yc) + 6].min(0) > 250)[0]
        return max(X0, (cols[0] if len(cols) else X0) + 10)
    paper = np.percentile(gray[Y0:Y1, X0:X1], 70)
    dark = gray < 0.8 * paper
    wipe = np.zeros((H, W), bool); wipe[Y0:Y1, X0:X1] = dark[Y0:Y1, X0:X1]
    ink = np.median(pg.rect[..., :3][wipe & (gray < 0.55 * paper)], axis=0) * 0.7
    f = ImageFont.truetype(TIMES, int(SIZE * 4))
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s{3,}", tp.text_of(40025)) if p.strip()]
    cv_ = Image.new("L", (W * 4, H * 4), 0); d = ImageDraw.Draw(cv_)
    d.text((X0 * 4, 355 * 4), "By Aloha Daily Staff", font=ImageFont.truetype(TIMES, 8 * 4), fill=235, anchor="lm")
    y, n = FIRST, 0
    for p in paras:
        cur, first = "", True
        words = p.split()
        for w in words + [None]:
            width = (X1 - left(y) - (INDENT if first else 0)) * 4
            t = (cur + " " + w) if (cur and w) else (w or cur)
            if w is not None and f.getlength(t) <= width:
                cur = t; continue
            if y > Y1 - 12:
                break
            d.text(((left(y) + (INDENT if first else 0)) * 4, y * 4), cur, font=f, fill=240, anchor="lm")
            y += PITCH; n += 1; first = False; cur = w or ""
        if y > Y1 - 12:
            break
    la = np.asarray(cv_.resize((W, H), Image.LANCZOS), np.float32) / 255
    la = np.clip(cv2.GaussianBlur(la, (0, 0), 0.45) * 1.5, 0, 1)
    out = tp.composite(pg, la, (Y0, Y1), ink, wipe_mask=wipe)
    Image.fromarray(out).save(dst)
    print(n, "lines")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
