"""crop_cmp.py GROUP K [DIR] [gain]: brightened 1:1 crops (y 100:356, x 100:356) of: upscaled original | out frame | hd_current frame -> DIR/crop_cmp.png"""
import sys, numpy as np, cv2
from pathlib import Path
from PIL import Image
g=sys.argv[1]; k=int(sys.argv[2]); D=Path(sys.argv[3]) if len(sys.argv)>3 else Path("out")/g
gain=float(sys.argv[4]) if len(sys.argv)>4 else 2.5
ld=lambda p: np.asarray(Image.open(p).convert("RGB"),np.float32)
o=cv2.resize(ld(f"data/{g}/orig/wtr{k:05d}.png"),(512,512),interpolation=cv2.INTER_CUBIC)
h=ld(D/f"wtr{k:05d}.png"); c=ld(f"data/{g}/hd_current/wtr{k:05d}.png")
cr=lambda a: np.clip(a[100:356,100:356]*gain,0,255)
m=np.concatenate([cr(o),cr(h),cr(c)],1).astype(np.uint8)
Image.fromarray(m).save(D/"crop_cmp.png")
