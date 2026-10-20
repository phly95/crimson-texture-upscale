"""headtex_compose.py HALF GOGSEED: compose the cphead front view from parts -> headtex/r_cphead_front_comp_<half>.png
base = best full-view redraw (s10, feature-aligned) mirrored to exact symmetry about x=512 (HALF = L|R: which half is copied), goggles = the vector-grounded goggle part pasted over the traced goggle area"""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_feat as FT, headtex_vecgog as VG, headtex_fin as F
T = F.T; D = FT.D
def compose(half="L", gseed=3, base="r_cphead_front_s10_fa6.png", tg="cphead", align=False):
    refimg = np.array(Image.open(D / f"v_{tg}_front.png").convert("RGB"))
    b0 = np.array(Image.open(D / base).convert("RGB"))
    if align: b0, _ = FT.align_field(refimg, b0, 6, 40)                                                          # outlines of goggles / strap / skin onto the traced ones
    b = b0.astype(np.float32)
    if half == "L": b[:, 512:] = b[:, 511::-1][:, :512]          # right half := mirror of the left half (columns 0..511 -> 1023..512)
    else: b[:, :512] = b[:, 1023:511:-1]
    x0, y0, x1, y1 = VG.CROP; g = np.array(Image.open(D / f"part_gogv_s{gseed}.png").convert("RGB")); g = cv2.resize(g, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32)
    # goggle mask = the vector padding shape (left+mirror+bridge), grown a little, feathered
    left, _ = VG.smooth_outline(); mirror = left * [-1, 1] + [1024, 0]; M = np.zeros(b.shape[:2], np.uint8)
    for p in (left, mirror): cv2.fillPoly(M, [np.round(p).astype(np.int32)], 255, cv2.LINE_AA)
    cv2.rectangle(M, (505, 342), (519, 440), 255, -1); M = cv2.dilate(M, np.ones((9, 9), np.uint8)); a = cv2.GaussianBlur(M.astype(np.float32) / 255, (0, 0), 3.0)[..., None]
    # align the goggle part to the vector outline (flow between the blurred dark-outline fields), so the rims sit on the traced shape
    vec = np.array(Image.open(D / "part_gogv_ref.png").convert("RGB")); gb = np.array(Image.open(D / f"part_gogv_s{gseed}.png").convert("RGB"))
    f = lambda im: cv2.GaussianBlur(((cv2.cvtColor(im, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) * 100 / 255) < 42).astype(np.float32), (0, 0), 5)
    u8 = lambda x: np.clip(x * 255, 0, 255).astype(np.uint8); fl = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(u8(f(vec)), u8(f(gb)), None)
    fl = cv2.GaussianBlur(fl, (0, 0), 10); m_ = np.hypot(fl[..., 0], fl[..., 1]); fl *= np.minimum(1, 60 / np.maximum(m_, 1e-6))[..., None]
    gx, gy = np.meshgrid(np.arange(gb.shape[1], dtype=np.float32), np.arange(gb.shape[0], dtype=np.float32)); gb = cv2.remap(gb, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    g = cv2.resize(gb, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32)
    full = b.copy(); full[y0:y1, x0:x1] = g; out = b * (1 - a) + full * a
    # the part's skin differs a little from the base around the goggles: pull the pasted goggles' broad tone to the base outside the mask
    ref = refimg / 255.
    out = np.clip(T.colour_lock(np.clip(out, 0, 255) / 255., ref, 10.0, 30.0), 0, 1)                           # broad tone back to the reference
    return (out * 255 + .5).astype(np.uint8)
if __name__ == "__main__":
    tg, base, g = sys.argv[1], sys.argv[2], int(sys.argv[3])
    for h in ("L", "R"): Image.fromarray(compose(h, g, base, tg, align=True)).save(D / f"r_{tg}_front_comp_{h}.png")
