import sys; sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit")
import regen as R, numpy as np, cv2
from PIL import Image
def fix(u, base="latest", band_px=44, white_thr=0.70, clean=17):
    ct = R.chosen(u, base); H, W = ct.shape[:2]
    og_rgba = np.asarray(Image.open(R.ROOT / "pipeline/units" / u / "orig.png").convert("RGBA"))
    og = cv2.resize(og_rgba, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    a_og = (og[..., 3] > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(a_og, cv2.DIST_L2, 5)
    L = og[..., :3].mean(-1)
    fr = ((L > white_thr) & (dist < band_px) & (a_og > 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fr)
    keep = np.zeros_like(fr)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > 0.02 * W * H * 0.2: keep[lab == i] = 1
    fr = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    A = ct.astype(np.float32) / 255; old = A[..., :3]
    # smooth, bright frame colour: grey closing removes dark marks, then a blur keeps the paper shading
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean, clean))
    fillc = np.dstack([cv2.GaussianBlur(cv2.morphologyEx(old[..., c], cv2.MORPH_CLOSE, k), (0, 0), 2.0) for c in range(3)])
    fillc = np.maximum(fillc, 0.90)
    m = cv2.GaussianBlur(fr.astype(np.float32), (0, 0), 1.2)[..., None]
    out = old * (1 - m) + fillc * m
    edge = ((A[..., 3] > 0.001) & (A[..., 3] < 0.999) & (cv2.dilate(fr, np.ones((9, 9), np.uint8)) > 0))[..., None]   # antialiased rim of the frame: white, not dark specks
    A[..., :3] = np.where(edge, fillc, np.where((A[..., 3] < 0.999)[..., None], old, out))
    return (A * 255 + .5).astype(np.uint8), fr
if __name__ == "__main__":
    from PIL import ImageDraw
    u = [x for x in R.last if x.endswith("150da4a0")][0]
    out, fr = fix(u)
    Image.fromarray(out, "RGBA").save(R.ROOT / "face_audit/hawaii_frame.png")
    b = R.chosen(u, "latest")
    sh = Image.new("RGB", (3 * 520, 520), (30, 30, 30))
    sh.paste(Image.fromarray(b[..., :3]).resize((510, 510)), (0, 0)); sh.paste(Image.fromarray(out[..., :3]).resize((510, 510)), (520, 0)); sh.paste(Image.fromarray(fr * 255).convert("RGB").resize((510, 510)), (1040, 0))
    sh.save(R.ROOT / "face_audit/hawaii_frame_cmp.png")
