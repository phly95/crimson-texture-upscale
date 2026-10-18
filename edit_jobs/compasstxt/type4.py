"""compasstxt aligned to the ENGINE's cuts (CSVM CompassTape.LabelSrc, authored 128x32 coordinates): the labels are cut out of the atlas by rectangles
N 84-96, NE 1-26, E 40-52, SE 27-52, S 52-64, SW 52-84, W 64-84, NW 84-116, so every letter must lie exactly inside its own cell:
N1 1-13 | E1 13-26 | S1 27-40 | E2 40-52 | S2 52-64 | W1 64-84 | N2 84-96 | W2 96-116. Letters are typeset into those cells (0.4 px side margins)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from fit import *
sys.path.insert(0, R + 'flux_fix'); import tol
from type2 import compose
S = 8
CELLS = [('N', 1, 13), ('E', 13, 26), ('S', 27, 40), ('E', 40, 52), ('S', 52, 64), ('W', 64, 84), ('N', 84, 96), ('W', 96, 116)]
M = 0.4
def render(path, e):
    mask = np.zeros((32 * S, 128 * S), np.float32)
    for ch, x0, x1 in CELLS:
        f = ImageFont.truetype(path, 400)
        im = Image.new('L', (900, 700), 0); ImageDraw.Draw(im).text((20, 20), ch, font=f, fill=255)
        arr = np.asarray(im); ys, xs = np.where(arr > 8)
        crop = Image.fromarray(arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
        wpx, hpx = int(round((x1 - x0 - 2 * M) * S)), int((Y1 - Y0) * S)
        m = np.asarray(crop.resize((wpx, hpx), Image.LANCZOS), np.float32) / 255
        if e > 0: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)))
        xo = int(round((x0 + M) * S)); mask[Y0 * S:Y0 * S + hpx, xo:xo + wpx] = np.maximum(mask[Y0 * S:Y0 * S + hpx, xo:xo + wpx], m[:, :wpx])
    return mask
if __name__ == '__main__':
    best = None
    for fn in ('oswald', 'bebas', 'barlow', 'league', 'fira'):
        path = FONTS.get(fn) or (R + 'button_agent/fonts/LeagueGothic-Regular.otf')
        try: ImageFont.truetype(path, 50)
        except OSError: continue
        for e in (0, 2, 4):
            im = compose(render(path, e), f'c4_{fn}_{e}.png'); mt = tol.metrics(R + 'pipeline/units/compasstxt__4a1a6169/orig.png', im)
            print(fn, e, mt['diff'], mt['dcov'], mt['iou50'])
            if best is None or mt['diff'] < best[0]: best = (mt['diff'], fn, e)
    print('best', best)
