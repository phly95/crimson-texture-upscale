"""edgefix.py NAME FINAL.png [--w 14]: FLUX draws a faint light/dark band along the outer border of a tile; at the picture's own outer border blend the outer W px back to the Lanczos base (s4/base.png)."""
import sys, numpy as np
from PIL import Image
from pathlib import Path
name, fin = sys.argv[1:3]; W = int(sys.argv[sys.argv.index('--w') + 1]) if '--w' in sys.argv else 32
base = np.asarray(Image.open(f'/home/deck/workspace/csvm-hd/tile_job/{name}/s4/base.png').convert('RGB')).astype(np.float32)
im = np.asarray(Image.open(fin).convert('RGB')).astype(np.float32)
H, Wd = im.shape[:2]
def ramp(n):
    r = np.zeros(n, np.float32); k = np.arange(W); r[:W] = np.clip(1 - (k - 18) / (W - 18), 0, 1); r[n - W:] = np.maximum(r[n - W:], r[:W][::-1]); return r
w = np.maximum(ramp(H)[:, None], ramp(Wd)[None, :])[..., None]
Image.fromarray((im * (1 - w) + base * w).clip(0, 255).astype(np.uint8)).save(fin); print('edgefix', fin)
