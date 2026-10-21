"""typeset_pardon.py IN.png OUT.png: SB_13_01_RECORD (the Presidential Pardon). The title is already right; the body (4 lines of ~8 px type), the two lines under the signature and the fine print
at the foot are re-set from the game's strings in Courier New on the straightened page (typeset_page.Page / composite), wiping only the dark ink of the old lines inside their boxes.
The foot line has no string: 'Official Seal of the Nation of Hollywood' is written as plausible form furniture."""
import re, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import typeset_page as tp

COUR, COURB = str(tp.F / "cour.ttf"), str(tp.F / "courbd.ttf")
BODY = tp.STR[40057]
SIG = [tp.STR[40058] + ", " + tp.STR[40059].split("]", 1)[1].split("\n")[0], tp.STR[40059].split("\n")[1]]
BOX = [(128, 505, 700, 590), (128, 655, 700, 695), (190, 900, 600, 920)]


def run(src, dst):
    im = np.asarray(Image.open(src).convert("RGBA"))
    pg = tp.Page(im, 0.55, (0.6, 0.69))
    H, W = pg.H, pg.W
    gray = cv2.cvtColor(pg.rect[..., :3], cv2.COLOR_RGB2GRAY)
    dark = gray < 120
    wipe = np.zeros((H, W), bool)
    for (x0, y0, x1, y1) in BOX:
        wipe[y0:y1, x0:x1] = dark[y0:y1, x0:x1]
    ink = np.median(pg.rect[..., :3][wipe], axis=0) * 0.8
    cv_ = Image.new("L", (W * 4, H * 4), 0); d = ImageDraw.Draw(cv_)
    size = 15.0
    f = ImageFont.truetype(COUR, int(size * 4))
    words, ls, cur = re.sub(r"\s+", " ", BODY).strip().split(), [], "     "
    for w in words:
        t = (cur + " " + w) if cur.strip() else cur + w
        if len(t) <= 61:
            cur = t
        else:
            ls.append(cur); cur = w
    ls.append(cur)
    for i, ln in enumerate(ls):
        d.text(((170 if i == 0 else 136) * 4 - (0 if i else 0) - (0 if i else 5 * 0.6 * size * 4 * 0), (518 + 19 * i) * 4), ln.lstrip() if i == 0 else ln, font=f, fill=240, anchor="lm")
    for i, ln in enumerate(SIG):
        d.text((135 * 4, (666 + 19 * i) * 4), ln, font=ImageFont.truetype(COURB, int(15 * 4)), fill=240, anchor="lm")
    d.text((236 * 4, 910 * 4), "Official Seal of the Nation of Hollywood", font=ImageFont.truetype(COUR, int(12.8 * 4)), fill=225, anchor="lm")
    la = np.asarray(cv_.resize((W, H), Image.LANCZOS), np.float32) / 255
    la = np.clip(cv2.GaussianBlur(la, (0, 0), 0.45) * 1.7, 0, 1)
    out = tp.composite(pg, la, (500, 930), ink, wipe_mask=wipe)
    Image.fromarray(out).save(dst)
    print(len(ls), "body lines")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
