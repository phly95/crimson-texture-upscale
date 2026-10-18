"""compasstxt: real letterforms. The four pairs NE SE SW NW are typeset in a condensed grotesque, stretched into the original's column ranges and full height
(rows 1-31, as the original's tall letters), weight chosen so the coverage per pair matches the original; colour = the original's vertical gradient; black bar from the original."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
sys.argv = [sys.argv[0]]
from fit import *            # noqa (fit.py main guarded)
FONT = FONTS[sys.argv[0] if False else 'bebas']
S = 8
mask = np.zeros((32 * S, 128 * S), np.float32)
for text, x0, x1 in pairs:
    best = None
    for e in range(0, 14, 1):
        m = glyph_mask(FONT, text, x0, x1, 0, e)
        cov = float(reduce(m)[:, x0:x1].mean()); tgt = float(A[:, x0:x1].mean())
        if best is None or abs(cov - tgt) < best[0]:
            best = (abs(cov - tgt), e, m)
    print(text, 'embolden', best[1], 'cov err', round(best[0], 3)); mask = np.maximum(mask, best[2])
H, W = mask.shape
up = cv2.resize(A, (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 1)
bar = np.zeros_like(up); bar[:, 123 * S:] = up[:, 123 * S:]
let = mask.copy(); let[:, 122 * S:] = 0
alpha = np.maximum(let, bar)
rows = np.arange(H, dtype=np.float32) / S
B = np.interp(rows, [1.0, 28.0], [140, 197]).astype(np.float32)
rgb = np.zeros((H, W, 3), np.float32); rgb[..., 0], rgb[..., 1], rgb[..., 2] = 197, 194, B[:, None]
rgb *= (let >= bar)[..., None]
Image.fromarray(np.dstack([rgb, alpha * 255]).clip(0, 255).astype(np.uint8), 'RGBA').save('output_typed.png')
