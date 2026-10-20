"""headtex_warp.py TARGET PORTRAIT_SEED: slice a normal front portrait into the mesh's front layout -> headtex/r_<tg>_front_port.png
1) thin-plate warp on YuNet landmarks (eyes, nose, mouth corners) + crown + chin;  2) per-row horizontal scaling so the head fills the mesh silhouette exactly;  below the chin the existing redraw is kept"""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_score as S, headtex_fin as F
T = F.T; D = F.H.ROOT / "face_audit/headtex"
def widths(mask):
    ys = np.where(mask.any(1))[0]; w = np.zeros(mask.shape[0]); l = np.zeros(mask.shape[0]); r = np.zeros(mask.shape[0])
    for y in ys: xs = np.where(mask[y])[0]; l[y], r[y], w[y] = xs[0], xs[-1], xs[-1] - xs[0]
    return ys[0], ys[-1], w, l, r
def chin_row(mask):
    """first row after the widest one where the silhouette has narrowed to the neck's width (neck = median width of the bottom 30 rows)"""
    t, b, w, l, r = widths(mask); neck = np.median(w[b - 30:b]); ymax = int(np.argmax(w)); ys = np.where((np.arange(len(w)) > ymax) & (w > 0) & (w <= neck * 1.12))[0]; return int(ys[0]) if len(ys) else b

def portrait_mask(p):
    bg = np.median(np.concatenate([p[:20].reshape(-1, 3), p[:, :12].reshape(-1, 3)]), 0); d = np.abs(p.astype(np.float32) - bg).sum(2) > 60
    return cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)) > 0
def warp(tg, ps, base_file, sl=12.0, sab=40.0, mirror=True):
    ref = np.array(Image.open(D / f"v_{tg}_front.png").convert("RGB")); zb = np.load(D / f"v_{tg}_front_z.npy"); mesh = zb > -1e8
    por = np.array(Image.open(D / f"portrait_{tg}_s{ps}.png").convert("RGB")); base = np.array(Image.open(D / base_file).convert("RGB"))
    fr, fp = S.face(ref), S.face(por)
    if fr is None or fp is None: raise SystemExit("no face found: ref %s portrait %s" % (fr is not None, fp is not None))
    pm = portrait_mask(por); pt, pb, pw, pl, pr = widths(pm); mt, mb, mw, ml, mr = widths(mesh); chin_r = chin_row(mesh)
    # chin of the portrait: lowest point of the face = YuNet box bottom is unreliable, use mouth->chin proportion of the reference (mouth to chin / eye to mouth)
    eyeR, mouthR = fr["lm"][:2, 1].mean(), fr["lm"][3:, 1].mean(); eyeP, mouthP = fp["lm"][:2, 1].mean(), fp["lm"][3:, 1].mean()
    chin_p = mouthP + (chin_r - mouthR) * (mouthP - eyeP) / max(mouthR - eyeR, 1)
    src = np.vstack([fp["lm"], [[512, pt], [512, chin_p]]]); dst = np.vstack([fr["lm"], [[512, mt], [512, chin_r]]])           # ref position -> portrait position
    h, w = ref.shape[:2]; anchors = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [512, h - 1], [0, h // 2], [w - 1, h // 2]], float)
    k = RBFInterpolator(np.vstack([dst, anchors]), np.vstack([src, anchors]) - np.vstack([dst, anchors]), kernel="thin_plate_spline"); g = 8
    xs, ys = np.meshgrid(np.arange(0, w + g, g), np.arange(0, h + g, g)); off = k(np.c_[xs.ravel(), ys.ravel()].astype(float)).reshape(xs.shape + (2,)); off = cv2.resize(off.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)); w1 = cv2.remap(por, gx + off[..., 0], gy + off[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    pm1 = cv2.remap(pm.astype(np.float32), gx + off[..., 0], gy + off[..., 1], cv2.INTER_LINEAR) > 0.5
    _, _, w1w, w1l, w1r = widths(pm1)
    # per-row horizontal fit of the head to the mesh silhouette (rows from crown to chin), smoothed
    out = w1.copy(); sc = np.ones(h); cx = np.full(h, 512.0)
    for y in range(mt, chin_r):
        if w1w[y] > 10 and mw[y] > 10: sc[y] = mw[y] / w1w[y]; cx[y] = (w1l[y] + w1r[y]) / 2
    sc = cv2.GaussianBlur(sc.astype(np.float32)[:, None], (0, 0), 6)[:, 0]
    for y in range(mt, chin_r):
        xs_ = (np.arange(w) - (ml[y] + mr[y]) / 2) / sc[y] + cx[y]; out[y] = cv2.remap(w1[y:y + 1], xs_.astype(np.float32)[None], np.full((1, w), 0, np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)[0]
    a = np.zeros((h, w), np.float32); a[mt:chin_r + 1] = 1; a = cv2.GaussianBlur(a, (0, 0), 6) * mesh
    # colour: broad tone to the reference, then composite over the existing redraw (neck)
    outc = np.clip(T.colour_lock(out / 255., ref / 255., sl, sab), 0, 1) * 255
    res = (base * (1 - a[..., None]) + outc * a[..., None])
    if mirror: res[:, 512:] = res[:, 511::-1][:, :512]                                                          # exact left-right symmetry (the mesh is symmetric)
    Image.fromarray(np.clip(res + .5, 0, 255).astype(np.uint8)).save(D / f"r_{tg}_front_port.png")
    m, fl = S.score_view(ref, np.clip(res + .5, 0, 255).astype(np.uint8), False); print("lowcorr %.3f  landmark errs (eyes,nose) %s" % (m["lowcorr"], m.get("lm_err_each")), "chin_ref", chin_r)
if __name__ == "__main__": warp(sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 12.0, float(sys.argv[5]) if len(sys.argv) > 5 else 40.0)
