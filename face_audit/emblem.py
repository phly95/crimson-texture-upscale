import sys; sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit"); sys.path.insert(0, "/home/deck/workspace/csvm-hd/tools")
import regen as R, numpy as np, cv2
from PIL import Image
import tile_upscale as tu
f = lambda s: [u for u in R.last if u.endswith(s)][0]
ua = f("d82b3acc"); og = R.og(ua)
OGBOX = (0, 391, 150, 541)
crop = Image.fromarray(og).crop(OGBOX)
ref = crop.resize((1024, 1024), Image.LANCZOS); ref.save("face_audit/emb/ref.png")
P = ("Redraw this small black and white heraldic emblem painted on the side of an aircraft as crisp, clean artwork, keeping exactly the same design: a knight's great helm seen from the front, "
     "a tall narrow helmet with a flat top, two narrow horizontal eye slits side by side in the middle and a small group of breathing holes on the lower left, a crest ornament on top, "
     "and two swords crossed behind the helmet, their pommels and crossguards showing at the left and at the upper right. White and light grey painted shapes with thick black outlines, flat graphic style. "
     "Keep the dark background, the metal fuselage panel and the stripe at the right exactly as in the reference. Do not draw a cross, do not draw a shield, do not add lettering.")
seeds = list(range(200, 212))
og_hp = lambda g: g - cv2.GaussianBlur(g, (0, 0), 6)
tgt = np.asarray(crop.convert("L"), np.float32) / 255
res = []
for s in seeds:
    out = f"face_audit/emb/s{s}.png"
    tu.locked_gen("face_audit/emb/ref.png", out, P, 1024, s)
    g = np.asarray(Image.open(out).convert("L").resize(crop.size, Image.BOX), np.float32) / 255
    a, b = og_hp(tgt), og_hp(g); sc = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))
    res.append((sc, s)); print("seed", s, "corr %.3f" % sc, flush=True)
print(sorted(res, reverse=True)[:5])
