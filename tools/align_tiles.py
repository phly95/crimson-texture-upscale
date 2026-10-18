"""align_tiles.py NAME STAGE ROWA COLA ROWB COLB [--rows y0:y1]: measure the sub-pixel shift between two overlapping tiles (out_), optionally write the shifted tile B (out_B.noshift.png keeps the old)."""
import sys, json, numpy as np, cv2
from pathlib import Path
from PIL import Image
name, stage = sys.argv[1:3]
a_id, b_id = sys.argv[3], sys.argv[4]          # e.g. 00 01
y0, y1 = (int(v) for v in (sys.argv[5] if len(sys.argv) > 5 else "0:300").split(":"))
apply = "--apply" in sys.argv
d = Path('/home/deck/workspace/csvm-hd/tile_job') / name / stage
g = json.loads((d / 'grid.json').read_text()); T, xs, ys = g['T'], g['xs'], g['ys']
ia, ib = (int(a_id[0]), int(a_id[1])), (int(b_id[0]), int(b_id[1]))
A = np.asarray(Image.open(d / f'out_{a_id}.png').convert('L'), np.float32)
B = np.asarray(Image.open(d / f'out_{b_id}.png').convert('L'), np.float32)
ox = xs[ib[1]] - xs[ia[1]]; oy = ys[ib[0]] - ys[ia[0]]
# overlap region in A coordinates: x ox..T ; B coordinates: 0..T-ox
wa = A[y0 + oy:y1 + oy, ox:T] if oy == 0 else None
wa = A[y0:y1, ox:T]; wb = B[y0 - oy:y1 - oy, 0:T - ox] if oy else B[y0:y1, 0:T - ox]
# use central part
c = slice(40, wa.shape[1] - 40)
sh, resp = cv2.phaseCorrelate(wa[:, c].astype(np.float64), wb[:, c].astype(np.float64))
print(f"{name} {stage} tile {b_id} vs {a_id}: shift (dx,dy)=({sh[0]:.2f},{sh[1]:.2f}) response {resp:.2f}")
if apply:
    import shutil
    shutil.copy(d / f'out_{b_id}.png', d / f'out_{b_id}.noshift.png')
    arr = np.asarray(Image.open(d / f'out_{b_id}.png').convert('RGB'))
    M = np.float32([[1, 0, -sh[0]], [0, 1, -sh[1]]])
    out = cv2.warpAffine(arr, M, (T, T), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    Image.fromarray(out).save(d / f'out_{b_id}.png'); print("applied")
