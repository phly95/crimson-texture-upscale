import sys, numpy as np, cv2
sys.path.insert(0, '/home/deck/workspace/csvm-hd/tools/pipeline')
from greenkey import green_key
from PIL import Image

RB = 12
ERODE = 18
RATIO = 4.0
CL = 25


def hybrid(rgb, k_strict=None, k_len=85):
    s, G, k0 = green_key(rgb, k0=k_strict)                    # shadows: neutral, translucent
    l, _, _ = green_key(rgb, k0=k_len)                        # lenient: keeps green paint
    a_s, a_l = s[..., 3] / 255.0, l[..., 3] / 255.0
    m = (a_s >= 0.9).astype(np.uint8)                          # solid in the strict key: car bodies (and black shadow cores); translucent shadow is not
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CL, CL)))
    ff = m.copy(); h, w_ = m.shape; msk = np.zeros((h + 2, w_ + 2), np.uint8); cv2.floodFill(ff, msk, (0, 0), 2)
    m = ((m == 1) | (ff == 0)).astype(np.uint8)                # fill holes
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 1500:
            keep[lab == i] = 1
    keep = cv2.erode(keep, np.ones((3, 3), np.uint8))
    ratio = rgb[..., 1] / (rgb[..., 0] + rgb[..., 2] + 1.0)                                 # the screen is ~20:1 green to red+blue; a shadow on it keeps a high ratio, paint does not
    paint = ((ratio < RATIO) & (keep > 0)).astype(np.uint8)         # inside the car mask only pixels with some red/blue are paint; pure-green-scaled ones are shadow
    paint = cv2.morphologyEx(paint, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    paint = cv2.morphologyEx(paint, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) & keep
    if ERODE:
        paint = cv2.erode(paint, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE + 1, 2 * ERODE + 1)))
    keep = cv2.dilate(paint, np.ones((3, 3), np.uint8))
    w = np.clip(cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.5), 0, 1)[..., None]
    col = s[..., :3].astype(np.float32) * (1 - w) + l[..., :3].astype(np.float32) * w
    a = a_s[..., None] * (1 - w) + np.maximum(a_s, a_l)[..., None] * w
    cap = np.maximum(col[..., 0], col[..., 2]) + 3.0                                         # outside the paint: no green cast in shadows or rims
    col[..., 1] = np.where(w[..., 0] > 0.5, col[..., 1], np.minimum(col[..., 1], cap))
    return np.dstack([np.clip(col, 0, 255), a * 255]).round().astype(np.uint8), keep

if __name__ == "__main__":
    U = 'pipeline/units/parklot1__2555a393/'
    def dark(im, w=240, h=480, c=(150, 150, 150)):
        im = im.convert('RGBA').resize((w, h), Image.LANCZOS); return Image.alpha_composite(Image.new('RGBA', (w, h), c + (255,)), im).convert('RGB')
    tiles = []
    for sub in ('retry/', ''):
        rgb = np.asarray(Image.open(U + sub + 'klein_s2_cm.png').convert('RGB')).astype(np.float32)
        arr, keep = hybrid(rgb); im = Image.fromarray(arr, 'RGBA'); im.save('vlm_work/parklot1_hyb_%s.png' % ('retry' if sub else 'first'))
        tiles += [dark(im), dark(im, c=(235, 235, 235)), Image.fromarray(keep * 255).resize((240, 480)).convert('RGB')]
    img = Image.new('RGB', (len(tiles) * 245, 480)); [img.paste(t, (i * 245, 0)) for i, t in enumerate(tiles)]; img.save('vlm_work/parklot1_hyb.png')
