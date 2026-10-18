import sys; sys.path.insert(0,"/home/deck/workspace/csvm-hd/face_audit")
import regen as R, numpy as np, cv2
from PIL import Image
from skimage import color as skc
def lift(u, box, base="latest", thr=0.30, sig=7.0, k=0.95):
    ct = R.chosen(u, base); H, W = ct.shape[:2]
    og = cv2.resize(R.og(u), (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    A = ct.astype(np.float32) / 255; old = A[..., :3]
    lo, lt = skc.rgb2lab(og), skc.rgb2lab(old)
    Lo = cv2.GaussianBlur(lo[..., 0], (0, 0), 2.0) / 100
    m = np.zeros((H, W), np.float32); x0, y0, x1, y1 = [int(v * s) for v, s in zip(box, (W, H, W, H))]
    m[y0:y1, x0:x1] = (Lo[y0:y1, x0:x1] > thr).astype(np.float32)
    m = cv2.GaussianBlur(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), (0, 0), 6)
    d = cv2.GaussianBlur(lo[..., 0], (0, 0), sig) - cv2.GaussianBlur(lt[..., 0], (0, 0), sig)
    lt[..., 0] += np.maximum(d, 0) * k * m
    out = np.clip(skc.lab2rgb(lt), 0, 1).astype(np.float32)
    A[..., :3] = np.where((A[..., 3] < 0.999)[..., None], old, out)
    return (A * 255 + .5).astype(np.uint8)
if __name__ == "__main__":
    u = [x for x in R.last if x.endswith("425cff02")][0]
    out = lift(u, (0.05, 0.50, 0.55, 0.97))
    Image.fromarray(out, "RGBA").save("/home/deck/workspace/csvm-hd/face_audit/pin1_legs.png")
    b = R.chosen(u, "latest"); w, h = 1024, 1024
    def crop(im): return Image.fromarray(im[..., :3]).crop((int(.05 * w), int(.45 * h), int(.55 * w), int(.98 * h))).resize((500, 530), Image.LANCZOS)
    sh = Image.new("RGB", (1530, 530)); sh.paste(Image.fromarray(R.og(u)).resize((1024, 1024), Image.LANCZOS).crop((int(.05 * w), int(.45 * h), int(.55 * w), int(.98 * h))).resize((500, 530)), (0, 0)); sh.paste(crop(b), (510, 0)); sh.paste(crop(out), (1020, 0)); sh.save("/home/deck/workspace/csvm-hd/face_audit/legs_cmp2.png")
