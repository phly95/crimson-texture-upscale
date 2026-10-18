"""sprite_verify.py ORIG.png CAND.png [SHEET.png]: does a redrawn soft-alpha sprite line up with the original?

Prints one JSON line of metrics and writes a contact sheet (original | candidate, each on checker / black / white, then both alpha channels).
The candidate is compared with the original at the ORIGINAL's size (box downsample), so a candidate drawn at 4x passes only if it reduces to the original.
  alpha_mae   mean |alpha diff| at full original size (0..1)         pass < 0.04
  alpha_box32 the CSVM author's metric (32x32 box average)           pass < 0.06
  coverage    mean alpha  orig / cand                                 pass within 0.05
  iou50       silhouette IoU at alpha > .5                           pass > 0.85
  soft_lost   share of original soft-band px (.08<a<.92) the candidate has under .04   pass < 0.15
  de_mean     mean CIE76 dE of premultiplied-over-mid-grey colour     pass < 8
  green_under_clear  candidate keeps green-key colour under alpha 0   pass == 0
Exit code 0 when every metric passes, 1 otherwise.
"""
import json, sys
import numpy as np
from PIL import Image
from skimage import color

def load(p):
    return Image.open(p).convert("RGBA")

def over(im, bg):
    base = Image.new("RGBA", im.size, bg)
    base.alpha_composite(im)
    return base.convert("RGB")

def checker(size, n=8):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    c = (((xx // n) + (yy // n)) % 2 * 55 + 100).astype(np.uint8)
    return Image.fromarray(np.dstack([c, c, c])).convert("RGBA")

def main():
    o, c = load(sys.argv[1]), load(sys.argv[2])
    c_small = c.resize(o.size, Image.BOX) if c.size != o.size else c
    ao = np.asarray(o.getchannel("A"), np.float32) / 255
    ac = np.asarray(c_small.getchannel("A"), np.float32) / 255
    ob, cb = ao > .5, ac > .5
    soft = (ao > .08) & (ao < .92)
    def box32(a):
        return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((32, 32), Image.BOX), np.float32) / 255
    grey = (128, 128, 128, 255)
    lo, lc = (np.asarray(over(i, grey), np.float32) / 255 for i in (o, c_small))
    de = color.deltaE_cie76(color.rgb2lab(lo), color.rgb2lab(lc))
    cr = np.asarray(c, np.uint8)
    green = (cr[..., 3] == 0) & (cr[..., 1] > 200) & (cr[..., 0] < 80) & (cr[..., 2] < 80)
    m = {"alpha_mae": float(np.abs(ao - ac).mean()), "alpha_box32": float(np.abs(box32(ao) - box32(ac)).mean()),
         "coverage_orig": float(ao.mean()), "coverage_cand": float(ac.mean()),
         "iou50": float((ob & cb).sum() / max(1, (ob | cb).sum())),
         "soft_lost": float((soft & (ac < .04)).sum() / max(1, soft.sum())),
         "de_mean": float(de.mean()), "green_under_clear": int(green.sum())}
    ok = (m["alpha_mae"] < .04 and m["alpha_box32"] < .06 and abs(m["coverage_orig"] - m["coverage_cand"]) < .05
          and m["iou50"] > .85 and m["soft_lost"] < .15 and m["de_mean"] < 8 and m["green_under_clear"] == 0)
    m["pass"] = bool(ok)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}))
    if len(sys.argv) > 3:
        S = 3 if max(o.size) <= 128 else 1                          # enlarge small sprites so detail is visible
        W, H = o.size[0] * S, o.size[1] * S
        def tile(im):
            return im.resize((W, H), Image.NEAREST) if S > 1 else im.resize((W, H))
        ci = c.resize((W, H), Image.LANCZOS)
        rows = []
        for im in (tile(o), ci):
            cols = []
            for bg in (None, (0, 0, 0, 255), (255, 255, 255, 255)):
                b = checker((W, H)) if bg is None else Image.new("RGBA", (W, H), bg)
                b.alpha_composite(im); cols.append(b.convert("RGB"))
            rows.append(cols)
        a_row = [Image.fromarray((np.asarray(i.getchannel("A")))).convert("RGB") for i in (tile(o), ci)]
        diff = Image.fromarray((np.abs(np.asarray(tile(o).getchannel("A"), np.int16) - np.asarray(ci.getchannel("A"), np.int16))).astype(np.uint8)).convert("RGB")
        sheet = Image.new("RGB", (W * 3 + 20, H * 3 + 20), (30, 30, 30))
        for r, cols in enumerate(rows):
            for k, im in enumerate(cols):
                sheet.paste(im, (k * (W + 10), r * (H + 10)))
        for k, im in enumerate(a_row + [diff]):
            sheet.paste(im, (k * (W + 10), 2 * (H + 10)))
        sheet.save(sys.argv[3])
    sys.exit(0 if ok else 1)

main()
