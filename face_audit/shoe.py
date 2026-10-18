import sys; sys.path.insert(0,"/home/deck/workspace/csvm-hd/face_audit"); sys.path.insert(0,"/home/deck/workspace/csvm-hd/tools")
import regen as R, numpy as np
from PIL import Image
import tile_upscale as tu
f=lambda s:[u for u in R.last if u.endswith(s)][0]
u=f("05ea6514"); lat=R.chosen(u,"latest")
box=(230,780,550,1024); c=Image.fromarray(lat[...,:3]).crop(box)
W,H=1024,784; c.resize((W,H),Image.LANCZOS).save("face_audit/shoe/ref.png")
P=("Edit only the shoe on the foot. Change the strappy criss-cross sandal into a backless high-heeled mule: an open-back slipper with a single wide strap across the toes only, "
   "a tall slim heel in a translucent clear glittery material, no ankle strap, no criss-cross straps, the heel of the foot bare and visible. "
   "Keep the leg, ankle, foot, skin tone, the other leg, the white background, lighting and every other detail exactly the same. Black and white photograph.")
for s in (301,302,303,304,305,306):
    tu.locked_gen_wh("face_audit/shoe/ref.png",f"face_audit/shoe/s{s}.png",P,W,H,s); print("seed",s,flush=True)
