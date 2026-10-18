"""film.py GROUP [DIR] [gain]: DIR/film.png = frames 0..7 as 200x200 crops (y,x 150:350) brightened by gain, 4x2 grid with 2px gaps"""
import sys, numpy as np
from pathlib import Path
from PIL import Image
g=sys.argv[1]; D=Path(sys.argv[2]) if len(sys.argv)>2 else Path("out")/g; gain=float(sys.argv[3]) if len(sys.argv)>3 else 2.0
fs=[np.clip(np.asarray(Image.open(D/f"wtr{k:05d}.png").convert("RGB"),np.float32)[150:350,150:350]*gain,0,255).astype(np.uint8) for k in range(8)]
pad=np.full((200,3,3),255,np.uint8)
rows=[np.concatenate(sum([[f,pad] for f in fs[r*4:r*4+4]],[]),1) for r in range(2)]
Image.fromarray(np.concatenate(rows,0)).save(D/"film.png")
