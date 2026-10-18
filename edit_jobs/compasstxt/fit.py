import sys, itertools, json
import numpy as np, cv2
from PIL import Image, ImageFont, ImageDraw
R = '/home/deck/workspace/csvm-hd/'
o = np.asarray(Image.open(R + 'pipeline/units/compasstxt__4a1a6169/orig.png').convert('RGBA'), np.float32)
A = o[..., 3] / 255                      # 32 x 128
S = 8                                    # HD scale (target 1024 x 256)
col = A[:, :122].mean(0)
# gaps between the four pairs NE SE SW NW: local minima of the column profile
mins = [i for i in range(10, 118) if col[i] <= col[max(0, i - 6):i + 7].min() + 1e-6 and col[i] < 0.30]
print('minima', mins, [round(float(col[i]), 2) for i in mins])

pairs = [('NE', 0, 26), ('SE', 27, 51), ('SW', 52, 89), ('NW', 90, 118)]
FONTS = {n: R + 'button_agent/fonts/' + f for n, f in {
    'bebas': 'BebasNeue-Regular.ttf', 'oswald': 'Oswald.ttf', 'league': 'LeagueGothic-Regular.otf', 'anton': 'Anton-Regular.ttf', 'pathway': 'PathwayGothicOne-Regular.ttf',
    'barlow': 'BarlowCondensed-Bold.ttf', 'barlowsb': 'BarlowCondensed-SemiBold.ttf', 'fira': 'FiraSansCondensed-Bold.ttf', 'rajdhani': 'Rajdhani-Bold.ttf'}.items()}
Y0, Y1 = 1, 31                          # letter rows in the original (full height)


def glyph_mask(font, text, x0, x1, track, e):
    """HD coverage (256 x 1024, float 0..1) of `text` stretched into columns x0..x1 and rows Y0..Y1 of the original, letters spaced by `track` (px at 8x), emboldened by e (px at 8x)"""
    f = ImageFont.truetype(font, 400)
    im = Image.new('L', (4000, 700), 0); d = ImageDraw.Draw(im)
    x = 20
    for ch in text:
        d.text((x, 20), ch, font=f, fill=255)
        x += f.getlength(ch) + track * 6
    arr = np.asarray(im); ys, xs = np.where(arr > 8)
    crop = Image.fromarray(arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    W, H = int((x1 - x0) * S), int((Y1 - Y0) * S)
    crop = crop.resize((W, H), Image.LANCZOS)
    out = np.zeros((32 * S, 128 * S), np.float32)
    m = np.asarray(crop, np.float32) / 255
    if e > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)); m = cv2.dilate(m, k)
    out[Y0 * S:Y0 * S + m.shape[0], x0 * S:x0 * S + m.shape[1]] = m[:min(m.shape[0], out.shape[0] - Y0 * S), :]
    return out


def reduce(m):
    return cv2.resize(m, (128, 32), interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    best = {}
    for fn, path in FONTS.items():
        tot = 0; cfg = []
        for text, x0, x1 in pairs:
            bb = None
            for e in (0, 2, 4, 6, 8, 10):
                for tr in (0, 20):
                    m = reduce(glyph_mask(path, text, x0, x1, tr, e))
                    err = float(np.abs(m[:, x0:x1] - A[:, x0:x1]).mean())
                    if bb is None or err < bb[0]:
                        bb = (err, e, tr)
            tot += bb[0]; cfg.append((text, bb))
        best[fn] = (tot / 4, cfg)
        print(fn, round(tot / 4, 4), [(t, b[1], b[2], round(b[0], 3)) for t, b in cfg])
    json.dump({k: [v[0], [(t, b[1], b[2]) for t, b in v[1]]] for k, v in best.items()}, open('fit.json', 'w'))
