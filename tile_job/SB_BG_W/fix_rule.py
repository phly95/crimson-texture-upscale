# post step: the thin top rule comes out of FLUX as a comb of dashes in x<1730; replace it by a smooth line (blur along x only, contrast restored)
import cv2, numpy as np, sys
from PIL import Image
inp, out = sys.argv[1:3]
im = np.asarray(Image.open(inp).convert("RGB")).astype(np.float32)
y0, y1, xa, xb = 150, 240, 0, 1900
reg = im[y0:y1, :xb].copy()
g = reg.mean(2)
hb = cv2.GaussianBlur(reg, (0, 0), sigmaX=12, sigmaY=0.01)
hg = hb.mean(2)
# line centre per column: row of darkest blurred value, smoothed
cy = np.argmin(cv2.GaussianBlur(hg, (0, 0), sigmaX=1, sigmaY=1.5), axis=0).astype(np.float32)
cy = cv2.GaussianBlur(cy[None, :], (0, 0), 20)[0]
rows = np.arange(reg.shape[0])[:, None]
w = np.clip(1 - (np.abs(rows - cy[None, :]) - 5) / 4, 0, 1)[..., None]
w = w * np.clip((xb - 40 - np.arange(xb)[None, :]) / 40.0, 0, 1)[..., None]
bg = cv2.GaussianBlur(reg, (0, 0), sigmaX=12, sigmaY=12)
# paper background without the line: median over rows far from the line, approximated by blur of the region with the line rows masked
k = 1.0
sm = bg + k * (hb - bg)
sm = np.minimum(sm, bg)
res = reg * (1 - w) + sm * w
im[y0:y1, :xb] = res
Image.fromarray(im.clip(0, 255).astype(np.uint8)).save(out)
