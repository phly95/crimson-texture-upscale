"""zach_wordfit.py SEED: fit the redrawn word window onto the anchor's emblem crop (ECC on ink masks, affine), blend inside the dilated ink mask -> emblem_fixed_<seed>.png (emblem crop, 1024)"""
import sys; sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit")
import numpy as np, cv2
from PIL import Image
seed = int(sys.argv[1])
ref = np.asarray(Image.open("face_audit/zach/ref.png").convert("L"), np.float32) / 255
WB = (100, 440, 520, 860)
old = ref[WB[1]:WB[3], WB[0]:WB[2]]; old = cv2.resize(old, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
new = np.asarray(Image.open(f"face_audit/zach/word_s{seed}.png").convert("L"), np.float32) / 255
def ink(g, thr=0.12):
    bg = cv2.GaussianBlur(cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8)), (0, 0), 6)
    return ((g - bg) > thr).astype(np.float32)
# restrict to the word: right/lower part (the 'Hunters' diagonal), drop Fortune remnants at the top
reg = np.zeros((1024, 1024), np.float32); reg[250:1024, 0:1024] = 1
io, inn = ink(old) * reg, ink(new) * reg
a, b = cv2.GaussianBlur(io, (0, 0), 6), cv2.GaussianBlur(inn, (0, 0), 6)
W = np.eye(2, 3, dtype=np.float32)
try:
    _, W = cv2.findTransformECC(a, b, W, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 400, 1e-6), None, 9)
except cv2.error as e:
    print("ecc failed", e)
print(np.round(W, 3))
warp = lambda x: cv2.warpAffine(x, W, (1024, 1024), flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
newc = np.asarray(Image.open(f"face_audit/zach/word_s{seed}.png").convert("RGB"), np.float32) / 255
newc = np.dstack([warp(newc[..., c]) for c in range(3)]); inw = warp(inn)
m = cv2.dilate(np.maximum(io, inw), np.ones((15, 15), np.uint8)); m = cv2.GaussianBlur(m, (0, 0), 5)[..., None]
oldc = np.asarray(Image.open("face_audit/zach/ref.png").convert("RGB"), np.float32)[WB[1]:WB[3], WB[0]:WB[2]] / 255
oldc = cv2.resize(oldc, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
# tone-match the new window to the old one at a broad scale so the badge/metal does not change shade
lo, ln = cv2.GaussianBlur(oldc, (0, 0), 25), cv2.GaussianBlur(newc, (0, 0), 25)
newc = np.clip(newc + (lo - ln) * (1 - np.clip(m, 0, 1) * 0.0), 0, 1)
res = oldc * (1 - m) + newc * m
# back to emblem-crop coordinates
win = cv2.resize(res, (WB[2] - WB[0], WB[3] - WB[1]), interpolation=cv2.INTER_AREA)
full = np.asarray(Image.open("face_audit/zach/ref.png").convert("RGB"), np.float32) / 255
mw = cv2.resize(m[..., 0], (WB[2] - WB[0], WB[3] - WB[1]), interpolation=cv2.INTER_AREA)[..., None]
out = full.copy(); sl = (slice(WB[1], WB[3]), slice(WB[0], WB[2]))
out[sl] = full[sl] * (1 - mw) + win * mw
Image.fromarray((out * 255 + .5).astype(np.uint8)).save(f"face_audit/zach/emblem_fixed_{seed}.png")
Image.fromarray((np.dstack([np.clip(io * 255, 0, 255)] * 3)).astype(np.uint8)).save(f"face_audit/zach/ink_old.png")
