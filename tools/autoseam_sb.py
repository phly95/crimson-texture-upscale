"""autoseam_sb.py NAME YMAX: align row-0 tiles (phase correlation on overlaps, 01 to 00, 02 to 01, 03 to 02), then choose each x seam at the column of the overlap where the two tiles agree best
(min smoothed |A-B| over rows 0..YMAX), write seams_s4.json (y seams 860/1544 wide ramps) + copy seam_assemble2.py, assemble -> tile_job/NAME_4x_text.png"""
import sys, json, subprocess, shutil, numpy as np, cv2
from pathlib import Path
from PIL import Image
ROOT = Path('/home/deck/workspace/csvm-hd'); name = sys.argv[1]; ymax = int(sys.argv[2])
d = ROOT / 'tile_job' / name / 's4'
g = json.loads((d / 'grid.json').read_text()); T, xs = g['T'], g['xs']
PY = str(ROOT / 'venv/bin/python')
for k in range(1, 4):
    subprocess.run([PY, '-I', str(ROOT / 'tools/align_tiles.py'), name, 's4', f'0{k-1}', f'0{k}', f'0:{ymax}', '--apply'], cwd=ROOT)
seams = []
for k in range(1, 4):
    A = np.asarray(Image.open(d / f'out_0{k-1}.png').convert('L'), np.float32)[:ymax]
    B = np.asarray(Image.open(d / f'out_0{k}.png').convert('L'), np.float32)[:ymax]
    ox = xs[k] - xs[k - 1]; ov = T - ox         # overlap width
    diff = np.abs(A[:, ox:T] - B[:, 0:ov]).mean(0)
    # also penalise columns where there is a lot of ink (less sensitive to mismatch but riskier)
    sm = np.convolve(diff, np.ones(9) / 9, 'same')
    lo, hi = 24, ov - 24
    j = lo + int(np.argmin(sm[lo:hi])); pos = xs[k] + j
    print(f'seam {k}: x={pos} (overlap {xs[k]}..{xs[k]+ov}) cost {sm[j]:.2f} vs median {np.median(sm[lo:hi]):.2f}')
    seams.append([pos, 12])
cfg = {"y": [{"x0": 0, "x1": 99999, "seams": [[860, 80], [1544, 100]]}], "x": seams}
(ROOT / 'tile_job' / name / 'seams_s4.json').write_text(json.dumps(cfg))
shutil.copy(ROOT / 'tools/seam_assemble2.py', ROOT / 'tile_job' / name / 'seam_assemble.py')
subprocess.run([PY, '-I', str(ROOT / 'tile_job' / name / 'seam_assemble.py'), name, 's4', str(ROOT / 'tile_job' / f'{name}_4x_text.png')], cwd=ROOT)
