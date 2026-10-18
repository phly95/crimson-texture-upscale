"""tol.py: the practical alpha tolerance (agreed 2026-10-17): a redraw is good enough when, with both alpha channels box-averaged to 32x32 (the CSVM author's metric),
the mean absolute difference is <= 0.12 AND the coverage (mean alpha) differs by <= 0.15. 'Looks right in game' decides the rest."""
import numpy as np
from PIL import Image

DIFF_MAX, COV_MAX = 0.12, 0.15


def alpha(path_or_im):
    im = path_or_im if isinstance(path_or_im, Image.Image) else Image.open(path_or_im)
    return im.convert("RGBA").getchannel("A")


def box32(a):
    return np.asarray(a.resize((32, 32), Image.BOX), np.float32) / 255


def metrics(orig, cand):
    ao, ac = alpha(orig), alpha(cand)
    bo, bc = box32(ao), box32(ac)
    diff = float(np.abs(bo - bc).mean())
    co = float(np.asarray(ao, np.float32).mean() / 255)
    cc = float(np.asarray(ac, np.float32).mean() / 255)
    o_s = np.asarray(ao.resize(ac.size, Image.BOX), np.float32) / 255 > .5
    c_s = np.asarray(ac, np.float32) / 255 > .5
    iou = float((o_s & c_s).sum() / max(1, (o_s | c_s).sum()))
    return {"diff": round(diff, 3), "cov_orig": round(co, 3), "cov_cand": round(cc, 3), "dcov": round(cc - co, 3), "iou50": round(iou, 3),
            "pass": bool(diff <= DIFF_MAX and abs(cc - co) <= COV_MAX)}
