import sys; sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit")
import regen as R, numpy as np, cv2
from PIL import Image
def clean(u, region, base="latest", dark=0.62, body_open=15, grow=9):
    ct = R.chosen(u, base); H, W = ct.shape[:2]
    A = ct.astype(np.float32) / 255; old = A[..., :3]; L = old.mean(-1)
    x0, y0, x1, y1 = [int(v * s) for v, s in zip(region, (W, H, W, H))]
    bgL = cv2.GaussianBlur(cv2.morphologyEx(L, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8)), (0, 0), 12)
    d = ((bgL - L) > 0.10).astype(np.uint8)                                   # anything clearly darker than the backdrop around it
    sub = np.zeros_like(d); sub[y0:y1, x0:x1] = d[y0:y1, x0:x1]
    body = cv2.morphologyEx(sub, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (body_open, body_open)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(body); keep = np.zeros_like(body)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > 4000: keep[lab == i] = 1
    body = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    stray = (sub > 0) & (body == 0)
    # soft halo: mid-grey fuzz next to the hair edge also counts
    halo = ((bgL - L) > 0.035) & (cv2.dilate(body, np.ones((61, 61), np.uint8)) > 0) & (body == 0)
    stray = (stray | halo) & (A[..., 3] > 0.99)
    m = cv2.dilate(stray.astype(np.uint8), np.ones((5, 5), np.uint8))
    bgr = cv2.cvtColor((old * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    inp = cv2.cvtColor(cv2.inpaint(bgr, m * 255, 6, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    # keep film grain level: add the backdrop's own grain back
    g = (old - cv2.GaussianBlur(old, (0, 0), 3)); gm = cv2.GaussianBlur(((body == 0) & (sub == 0)).astype(np.float32), (0, 0), 8)
    rng = np.random.default_rng(0); noise = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.8)[..., None] * float(g[(body == 0) & (sub == 0)].std() if ((body == 0) & (sub == 0)).any() else 0.01) * 1.3
    mf = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3)[..., None]
    out = old * (1 - mf) + np.clip(inp + noise, 0, 1) * mf
    A[..., :3] = out
    return (A * 255 + .5).astype(np.uint8), m
if __name__ == "__main__":
    f = lambda s: [u for u in R.last if u.endswith(s)][0]
    u = f("3f15473c"); out, m = clean(u, (0.55, 0.0, 1.0, 0.5))
    Image.fromarray(out, "RGBA").save("face_audit/myshoe_clean.png")
    b = R.chosen(u, "latest"); H, W = b.shape[:2]
    def crop(im): return Image.fromarray(im[..., :3]).crop((int(.45 * W), int(.02 * H), int(.95 * W), int(.38 * H))).resize((600, int(600 * .36 * H / (.5 * W))), Image.LANCZOS)
    sh = Image.new("RGB", (1210, 600), (30, 30, 30)); sh.paste(crop(b), (0, 0)); sh.paste(crop(out), (610, 0)); sh.save("face_audit/myshoe_clean_cmp.png")
