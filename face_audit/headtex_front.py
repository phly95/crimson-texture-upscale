"""headtex_front.py TARGET VIEW SEED... : extra klein seeds for one view with a symmetric-face prompt -> headtex/r_<tg>_<view>_s<seed>.png"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_run as R
T, F, D = R.T, R.F, R.D
tg, k = sys.argv[1], sys.argv[2]; C = R.TARGETS[tg]
TAILS = {"woman": " The face is symmetrical and smiling naturally with teeth, both eyes open and level, a straight normal nose, smooth skin; the dark leather cap ends in one clean smooth line across the forehead with no seams or folds on the face.",
         "fbhead1": " The face is symmetrical and smiling gently, both eyes open and level behind round glasses, a straight normal nose; the white woolly cap ends in one clean line across the forehead."}
TAIL = TAILS.get(tg, " The face is symmetrical with a straight, normal nose and a relaxed neutral mouth; both round goggle lenses are the same size and sit side by side over the eyes; no strap hanging across the neck.") if k == "front" else ""
for seed in map(int, sys.argv[3:]):
    src = D / f"v_{tg}_{k}.png"; out = D / f"g_{tg}_{k}_s{seed}.png"
    T.locked_gen_wh(str(src), str(out), "This is " + C["who"] + R.COMMON + R.NOFACE.get(k, "") + TAIL, 1024, 1024, seed)
    ref = np.array(Image.open(src).convert("RGB")); gen = np.array(Image.open(out).convert("RGB").resize((1024, 1024), Image.LANCZOS))
    w, fl = F.align(gen, ref, smooth=14.0, clip=30.0); fn = np.clip(T.colour_lock(w / 255., ref / 255., 10.0, 30.0), 0, 1)
    Image.fromarray((fn * 255 + .5).astype(np.uint8)).save(D / f"r_{tg}_{k}_s{seed}.png"); print(seed, "lowcorr %.3f" % F.lowcorr((fn * 255).astype(np.uint8), ref), flush=True)
