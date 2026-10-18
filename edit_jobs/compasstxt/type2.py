import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0]] + sys.argv[1:]
import numpy as np, cv2
from fit import *
import fit
import glob
sys.path.insert(0, R + 'flux_fix'); import tol
S = 8
def build(font, es, tr=0):
    mask = np.zeros((32 * S, 128 * S), np.float32)
    for (text, x0, x1), e in zip(pairs, es):
        m = glyph_mask(font, text, x0, x1, tr, abs(e)) if e >= 0 else None
        if e < 0:
            m = glyph_mask(font, text, x0, x1, tr, 0)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (-e) + 1, 2 * (-e) + 1)); m = cv2.erode(m, k)
        mask = np.maximum(mask, m)
    return mask
def compose(mask, out):
    H, W = mask.shape
    up = cv2.resize(A, (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    bar = np.zeros_like(up); bar[:, 123 * S:] = up[:, 123 * S:]
    let = mask.copy(); let[:, 122 * S:] = 0
    alpha = np.maximum(let, bar)
    rows = np.arange(H, dtype=np.float32) / S
    B = np.interp(rows, [1.0, 28.0], [140, 197]).astype(np.float32)
    rgb = np.zeros((H, W, 3), np.float32); rgb[..., 0], rgb[..., 1], rgb[..., 2] = 197, 194, B[:, None]
    rgb *= (let >= bar)[..., None]
    im = Image.fromarray(np.dstack([rgb, alpha * 255]).clip(0, 255).astype(np.uint8), 'RGBA'); im.save(out); return im
if __name__ == '__main__':
    res = []
    for fn, path in FONTS.items():
        try: ImageFont.truetype(path, 50)
        except OSError: continue
        es = []
        for text, x0, x1 in pairs:
            bb = None
            for e in range(-8, 10, 2):
                for tr in (0,):
                    m = glyph_mask(path, text, x0, x1, tr, e) if e >= 0 else cv2.erode(glyph_mask(path, text, x0, x1, tr, 0), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * -e + 1, 2 * -e + 1)))
                    err = float(np.abs(reduce(m)[:, x0:x1] - A[:, x0:x1]).mean())
                    if bb is None or err < bb[0]: bb = (err, e)
            es.append(bb[1])
        im = compose(build(path, es), f'cand_{fn}.png'); mt = tol.metrics(R + 'pipeline/units/compasstxt__4a1a6169/orig.png', im)
        res.append((mt['diff'], fn, es, mt)); print(fn, es, mt)
    res.sort(); print('best', res[0][:3])
