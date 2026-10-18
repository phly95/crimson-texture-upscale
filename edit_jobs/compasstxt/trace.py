"""compasstxt: crisp HD version made by tracing the original's own letter shapes (upscale the low-res alpha, smooth, threshold) instead of redrawing them.
Letters: alpha = smoothstep around a threshold chosen so the letter coverage equals the original's; colour = the original's vertical gradient (197,194,B) with B 140 -> 197.
The black bar on the right (cols 123-127) is the original's, smoothly upscaled."""
import numpy as np, cv2
from PIL import Image
R = '/home/deck/workspace/csvm-hd/'
S = 8
o = np.asarray(Image.open(R + 'pipeline/units/compasstxt__4a1a6169/orig.png').convert('RGBA'), np.float32) / 255
A = o[..., 3]
W, H = 128 * S, 32 * S
LET = 122 * S                                          # letters occupy the columns left of the bar's gap
up = cv2.resize(A, (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 1)
sm = cv2.GaussianBlur(up, (0, 0), 0.55 * S)
target = float(A[:, :122].mean())

def crisp(t, w=0.035):
    x = np.clip((sm - (t - w)) / (2 * w), 0, 1)
    return x * x * (3 - 2 * x)

lo, hi = 0.05, 0.95
for _ in range(30):
    t = (lo + hi) / 2
    c = float(cv2.resize(crisp(t)[:, :LET], (122, 32), interpolation=cv2.INTER_AREA).mean())
    lo, hi = (t, hi) if c > target else (lo, t)
t = (lo + hi) / 2
let = crisp(t); let[:, LET:] = 0
bar = np.zeros_like(up); bar[:, 123 * S:] = up[:, 123 * S:]
alpha = np.maximum(let, bar)
rows = np.arange(H, dtype=np.float32) / S
B = np.interp(rows, [1.0, 28.0], [140, 197]).astype(np.float32)
rgb = np.zeros((H, W, 3), np.float32)
rgb[..., 0], rgb[..., 1], rgb[..., 2] = 197, 194, B[:, None]
rgb *= (let > 0.0)[..., None] * (let >= bar)[..., None]            # the bar stays black
out = np.dstack([rgb, alpha * 255]).clip(0, 255).astype(np.uint8)
Image.fromarray(out, 'RGBA').save('output.png')
print('threshold', round(t, 3), 'target coverage', round(target, 3))
