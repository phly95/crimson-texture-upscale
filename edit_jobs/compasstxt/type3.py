import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from fit import *
sys.path.insert(0, R + 'flux_fix'); import tol
from type2 import compose
S = 8
boxes = [('N', 0, 13), ('E', 14, 26), ('S', 27, 38), ('E', 39, 50), ('S', 51, 64), ('W', 65, 91), ('N', 91, 104), ('W', 104, 118)]
def render(font, e):
    mask = np.zeros((32 * S, 128 * S), np.float32)
    for ch, x0, x1 in boxes:
        mask = np.maximum(mask, glyph_mask(font, ch, x0, x1, 0, e))
    return mask
best = None
for fn in ('bebas', 'anton', 'barlow', 'oswald', 'league'):
    path = FONTS.get(fn) or (R + 'button_agent/fonts/LeagueGothic-Regular.otf')
    try: ImageFont.truetype(path, 50)
    except OSError: continue
    for e in (0, 2, 4, 6):
        m = render(path, e); im = compose(m, f'c3_{fn}_{e}.png'); mt = tol.metrics(R + 'pipeline/units/compasstxt__4a1a6169/orig.png', im)
        print(fn, e, mt['diff'], mt['dcov'], mt['iou50'])
        if best is None or mt['diff'] < best[0]: best = (mt['diff'], fn, e)
print('best', best)
