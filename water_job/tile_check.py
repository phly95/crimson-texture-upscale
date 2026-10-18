"""tile_check.py GROUP [DIR]: writes DIR/tile_check.png (2x2 tiled frame 0 | 2x2 tiled frame 3) and tile_check_bright.png (same, brightened x2.5 for inspection)."""
import sys, numpy as np
from pathlib import Path
from PIL import Image
g=sys.argv[1]; D=Path(sys.argv[2]) if len(sys.argv)>2 else Path("out")/g
ims=[]
for k in (0,3):
    a=np.asarray(Image.open(D/f"wtr{k:05d}.png").convert("RGB"))
    ims.append(np.tile(a,(2,2,1)))
t=np.concatenate(ims,1)
Image.fromarray(t).resize((t.shape[1]//2,t.shape[0]//2),Image.LANCZOS).save(D/"tile_check.png")
Image.fromarray(t).save(D/"tile_check_full.png")
b=np.clip(t.astype(np.float32)*2.5,0,255).astype(np.uint8)
Image.fromarray(b).resize((b.shape[1]//2,b.shape[0]//2),Image.LANCZOS).save(D/"tile_check_bright.png")
