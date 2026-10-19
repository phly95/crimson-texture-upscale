import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit")
import regen as R, numpy as np
from PIL import Image
f = lambda s: [u for u in R.last if u.endswith(s)][0]
ua = f("3ad54545")
for t, c in (("6b4b3dd2", (0.62, 0.38)), ("d947fbd6", (0.63, 0.46))):
    out, msg = R.regen(ua, f(t), c, k=(1.2, 1.25), shift=-0.1)
    print(t, msg)
    Image.fromarray(out, "RGBA").save(f"face_audit/zach/head_{t}.png")
