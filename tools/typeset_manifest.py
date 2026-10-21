"""typeset_manifest.py IN.png OUT.png: SB_08_01_MANIFEST's printed Russian form.
The original prints Latin-looking nonsense in a Cyrillic style, and a redraw cannot keep it from turning into noise. The form text is re-set as real, plausible Russian
(title, ribbon, four header lines, a three-row cargo table and the fine print under the torn note) in Times, wiping only the teal ink of the old text and leaving the rules, the
stamp and the note paper alone. The handwritten note is set by typeset_page.typeset_blocks first (run it, then this on its output)."""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import typeset_page as tp

TIMES, TIMESI, TIMESB = (str(tp.F / n) for n in ("times.ttf", "Timesi.TTF", "TIMESBD.TTF"))
HEAD = (10, 12, 28)                                            # title and ribbon ink
FORM = (14, 62, 84)                                            # the teal of the lines and the form text

# (text, x, y_centre, size px, font, ink, anchor), in the straightened frame
ITEMS = [
    ("МАНИФЕСТ", 404, 171, 40, TIMES, HEAD, "mm"),
    ("ОТПРАВКА", 388, 240, 24, TIMES, HEAD, "mm"),
    ("Накладная: 18287", 100, 347, 24, TIMESI, FORM, "lm"),
    ("Отправитель: Ф. Фассенбиндер", 100, 373, 24, TIMESI, FORM, "lm"),
    ("Номер рейса: РЦ-82", 100, 398, 24, TIMESI, FORM, "lm"),
    ("Назначение: Нью-Йорк, пирс 9", 100, 422, 24, TIMESI, FORM, "lm"),
    ("Наименование", 100, 477, 14, TIMESI, FORM, "lm"),
    ("Кол-во", 432, 477, 14, TIMESI, FORM, "lm"),
    ("Нитро-ускорители", 102, 537, 15, TIMESI, FORM, "lm"),
    ("30 (по 80 кг)", 452, 537, 15, TIMESI, FORM, "lm"),
    ("Топливные насосы", 102, 596, 15, TIMESI, FORM, "lm"),
    ("14", 452, 596, 15, TIMESI, FORM, "lm"),
    ("Оборудование (тип «К»)", 102, 653, 15, TIMESI, FORM, "lm"),
]
FINE = [("Груз подлежит досмотру. Вскрытие упаковки без разрешения отправителя запрещено.", 170, 927)]
OLD = [(290, 146, 520, 194), (300, 228, 480, 254), (96, 336, 560, 361), (96, 362, 560, 387), (96, 388, 560, 411), (96, 411, 560, 434),
       (92, 466, 200, 490), (420, 466, 500, 490), (92, 522, 330, 552), (440, 522, 700, 552), (92, 580, 330, 610), (440, 580, 700, 610), (92, 636, 330, 662), (335, 640, 430, 664)]


def tear(g, y0=800, y1=990):
    """per column the first row below the torn note (the note is bright paper, the strip under it is dark ink on grey)"""
    H, W = g.shape
    out = np.full(W, y0, int)
    sm = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 2.0)
    for x in range(W):
        col = sm[y0:y1, x]
        bright = col > 172
        y = 0
        while y < len(col) and bright[y]:
            y += 1
        out[x] = y0 + y
    return cv2.medianBlur(out.astype(np.uint8) if out.max() < 256 else out.astype(np.int32).astype(np.float32), 5) if False else out


def run(src, dst):
    im = np.asarray(Image.open(src).convert("RGBA"))
    pg = tp.Page(im, 0.6, (0.45, 0.66))
    H, W = pg.H, pg.W
    rgb = pg.rect[..., :3].astype(int)
    gray = cv2.cvtColor(pg.rect[..., :3], cv2.COLOR_RGB2GRAY)
    teal = (gray < 150) & (rgb[..., 2] >= rgb[..., 0] + 4)       # teal or dark ink, not the red stamp
    tz = tear(gray)
    # the table and frame rules are teal too: only wipe the thin text strokes, i.e. inside each old text box and away from long horizontal runs
    long_h = cv2.morphologyEx(teal.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))) > 0
    long_h = cv2.dilate(long_h.astype(np.uint8), np.ones((3, 1), np.uint8)) > 0
    wipe = np.zeros((H, W), bool)
    for (x0, y0, x1, y1) in OLD:
        wipe[y0:y1, x0:x1] = (teal & ~long_h)[y0:y1, x0:x1]
    # the fine print: strip pixels below the tear
    cols = np.arange(W)[None, :]; rows = np.arange(H)[:, None]
    below = rows > (tz[None, :] + 3)
    fine = np.zeros((H, W), bool); fine[900:946, 90:735] = True
    wipe |= (fine & below & teal & ~long_h)
    canvases = {}
    def canvas(ink):
        k = tuple(ink)
        if k not in canvases:
            canvases[k] = Image.new("L", (W * 4, H * 4), 0)
        return canvases[k]
    for text, x, y, size, fontp, ink, anchor in ITEMS:
        ImageDraw.Draw(canvas(ink)).text((x * 4, y * 4), text, font=ImageFont.truetype(fontp, int(size * 4)), fill=240, anchor=anchor)
    for text, x, y in FINE:
        ImageDraw.Draw(canvas(FORM)).text((x * 4, y * 4), text, font=ImageFont.truetype(TIMESI, 11 * 4), fill=235, anchor="lm")
    out = pg.rect[..., :3].astype(np.float32)
    old = cv2.dilate(wipe.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    keep = (~old & (gray > 120)).astype(np.float32)               # neither wiped ink nor the dark tear shadow feeds the fill
    smooth = cv2.GaussianBlur(out * keep[..., None], (0, 0), 3) / np.maximum(cv2.GaussianBlur(keep, (0, 0), 3), 1e-3)[..., None]
    noise = cv2.GaussianBlur(np.random.default_rng(3).normal(0, 1.0, (H, W)).astype(np.float32), (0, 0), 0.8)[..., None] * 2.0
    out = np.where(old[..., None], smooth + noise, out)
    for ink, cv in canvases.items():
        la = np.asarray(cv.resize((W, H), Image.LANCZOS), np.float32) / 255
        la = np.clip(cv2.GaussianBlur(la, (0, 0), 0.45) * 1.25, 0, 1)
        if ink == FORM:
            la = la * np.where(rows > 880, below.astype(np.float32), 1.0)     # fine print only below the tear
        out = out * (1 - la[..., None]) + np.asarray(ink, np.float32)[None, None, :] * la[..., None]
    rect2 = pg.rect.copy(); rect2[..., :3] = out.clip(0, 255).astype(np.uint8)
    # back to the page's frame: only where text or wiped ink is, so nothing else is resampled
    m = (old | (np.max([np.asarray(c.resize((W, H), Image.BOX)) for c in canvases.values()], axis=0) > 0)).astype(np.uint8) * 255
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    mp = cv2.warpAffine(m, pg.Mi, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255
    mp = cv2.GaussianBlur(mp, (0, 0), 1.0)[..., None] if mp.ndim == 2 else cv2.GaussianBlur(mp[..., 0], (0, 0), 1.0)[..., None]
    back = cv2.warpAffine(rect2[..., :3], pg.Mi, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    res = im.copy()
    res[..., :3] = (im[..., :3] * (1 - mp) + back * mp).clip(0, 255).astype(np.uint8)
    Image.fromarray(res).save(dst)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
