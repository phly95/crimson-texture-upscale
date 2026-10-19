import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit")
import regen as R, numpy as np
from PIL import Image
Z = "/home/deck/workspace/csvm-hd/face_audit/zach/"
f = lambda s: [u for u in R.last if u.endswith(s)][0]
ua = f("3ad54545"); orig_chosen = R.chosen
anc = np.asarray(Image.open(Z + "final_3ad54545.png").convert("RGBA"))
R.chosen = lambda u, base="chosen": anc if u == ua else orig_chosen(u, base)
out, msg = R.recompose(ua, f("d947fbd6"), base="latest", anchor_base="latest")
print(msg); Image.fromarray(out, "RGBA").save(Z + "final_d947fbd6_recomp.png")
