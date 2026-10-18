"""blackfix.py NAME IN.png OUT.png: restore pure-black regions (source pixels exactly 0, connected areas > 1000 px at source size) in a redraw of tile_job/NAME/src.png."""
import sys, cv2, numpy as np
from PIL import Image
from pathlib import Path
name, i, o = sys.argv[1:4]
src = np.asarray(Image.open(Path('/home/deck/workspace/csvm-hd/tile_job') / name / 'src.png').convert('RGB')).max(2)
m = (src == 0).astype(np.uint8); n, l, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
keep = np.zeros_like(m)
for k in range(1, n):
    if st[k, 4] > 1000: keep[l == k] = 1
im = np.asarray(Image.open(i).convert('RGB')).astype(np.float32)
H, W = im.shape[:2]
mk = cv2.resize(keep.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
mk = cv2.GaussianBlur(mk, (0, 0), W / 800 * 0.5)
mk = np.clip((mk - 0.05) / 0.9, 0, 1)
Image.fromarray((im * (1 - mk[..., None])).clip(0, 255).astype(np.uint8)).save(o); print('wrote', o)
