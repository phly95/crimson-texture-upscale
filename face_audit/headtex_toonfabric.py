"""headtex_toonfabric.py: cartoon (flat painted, soft cel-shaded) version of a small fabric UV texture IN ITS EXACT LAYOUT without a generative redraw (klein turns these
32x64 fabric crops into whole garments): smooth upscale -> mean-shift flattening -> soft lightness banding; plus part-specific vector details.
womanchest: neckline skin recoloured to the woman's skin (headtex/woman_skin_target.npy) with a clean dark neckline outline; grey atlas background painted as red fabric.
woman1: the belt band along the top repainted as a clean dark-brown belt with tan buckle bits where the original has them."""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_figure as FG
D = FG.ROOT / "face_audit/headtex"
def flat(img, W, H, sig=None, bands=3, soft=0.5, keep=0.35):
    """flat cartoon fabric: one flat colour (median chroma), lightness = the original's broad shading only (heavy blur), softly banded into cel steps"""
    up = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC); lab = cv2.cvtColor(up, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = cv2.GaussianBlur(lab[..., 0], (0, 0), sig or W / 28); lo, hi = np.percentile(L, 3), np.percentile(L, 97)
    t = np.clip((L - lo) / max(hi - lo, 1), 0, 1) * (bands - 1); f = np.floor(t); r = t - f
    st = np.clip((r - (0.5 - soft / 2)) / soft, 0, 1); st = st * st * (3 - 2 * st); q = (f + st) / (bands - 1) * (hi - lo) + lo
    out = lab.copy(); out[..., 0] = (1 - keep) * q + keep * L; out[..., 1:] = cv2.GaussianBlur(lab[..., 1:], (0, 0), W / 16) * 0.5 + np.median(lab[..., 1:].reshape(-1, 2), 0) * 0.5
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
def outline(img, mask, px, col=(28, 12, 10)):
    m = mask.astype(np.uint8); edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))).astype(np.float32)
    a = cv2.GaussianBlur(edge, (0, 0), px / 4)[..., None]; return (img * (1 - a) + np.array(col, np.float32) * a).astype(np.uint8)
def chest(W=512, H=1024):
    o = FG.orig("womanchest"); up = cv2.resize(o, (W, H), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(up, cv2.COLOR_RGB2LAB).astype(np.float32); A, B, L = lab[..., 1] - 128, lab[..., 2] - 128, lab[..., 0]
    skin = (B > 18) & (L > 90); skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)); skin = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), W / 24) > 0.5                      # smooth neckline curve (the 32x64 mask is blocky)
    grey = (np.hypot(A, B) < 10) & (L > 60); n, lbl = cv2.connectedComponents(grey.astype(np.uint8)); grey = np.isin(lbl, list(set(np.unique(np.r_[lbl[:, 0], lbl[-1]])) - {0}))
    grey = cv2.GaussianBlur(cv2.dilate(grey.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 4) > 0.5
    fab = ~skin & ~grey; red = np.median(up[fab].reshape(-1, 3), 0)
    src = up.copy(); src[grey] = red                                                               # atlas background -> red fabric before flattening
    out = flat(src, W, H)
    arm = np.array(Image.open(FG.ROOT / "pipeline/units/womanarm__d555fe04/custom/up2.png").convert("RGB")); al = cv2.cvtColor(arm, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    sleeve = al[(al[:, 1] > 150) & (al[:, 2] > 140) & (al[:, 1] - 128 > 1.3 * (al[:, 2] - 128))]; sm = np.median(sleeve, 0)                 # the red sleeve of the submitted womanarm
    ol = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32); fm = np.median(ol[fab], 0); sh = (sm - fm) * np.array([0.7, 1.0, 1.0], np.float32)
    w = cv2.GaussianBlur(fab.astype(np.float32) + grey.astype(np.float32), (0, 0), 2)[..., None]; out = cv2.cvtColor(np.clip(ol + w * sh, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    print("sleeve Lab", sm.round(1), "chest fabric Lab before", fm.round(1))
    import headtex_skin as SK; st = SK.mean_skin(np.array(Image.open(FG.ROOT / "pipeline/units/womanarm__d555fe04/custom/up2.png").convert("RGB"))).astype(np.float32); sl = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    sk = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), 2)[..., None]; tgt = sl.copy(); tgt[..., 1:] = st[1:]; tgt[..., 0] = st[0] + (sl[..., 0] - sl[..., 0][skin].mean()) * 0.6
    out = cv2.cvtColor(np.clip(sl * (1 - sk) + tgt * sk, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    out = outline(out, skin, max(5, W // 70)); return out, dict(skin_frac=float(skin.mean()), grey_frac=float(grey.mean()), red=red.tolist())
def trousers(W=512, H=1024):
    o = FG.orig("woman1"); up = cv2.resize(o, (W, H), interpolation=cv2.INTER_CUBIC); out = flat(up, W, H)
    oh = o.shape[0]; lab = cv2.cvtColor(o, cv2.COLOR_RGB2LAB).astype(int)
    rows = [y for y in range(min(8, oh)) if (lab[y, :, 0] < 70).mean() > 0.5]                        # dark belt rows of the 32x64 original
    if rows:
        y0, y1 = min(rows) * H // oh, (max(rows) + 1) * H // oh; band = out[y0:y1].astype(np.float32)
        belt = np.array((40, 30, 22), np.float32); out[y0:y1] = (belt * (0.9 + 0.25 * np.linspace(1, 0, y1 - y0)[:, None, None])).astype(np.uint8)   # near-black brown (these rows also wrap onto the cuffs)
        tan = (lab[min(rows):max(rows) + 1, :, 2] > 150) & (lab[min(rows):max(rows) + 1, :, 0] > 90); cols = np.where(tan.any(0))[0]
        if len(cols):                                                                              # one clean buckle frame over the light columns (+ its prong)
            x0, x1 = cols.min() * W // o.shape[1], (cols.max() + 1) * W // o.shape[1]; th = max(4, W // 90); pad = max(4, (y1 - y0) // 8)
            cv2.rectangle(out, (x0 + th, y0 + pad), (x1 - th - 1, y1 - pad - 1), (190, 152, 84), th, cv2.LINE_AA); cv2.line(out, ((x0 + x1) // 2, y0 + pad), ((x0 + x1) // 2, y1 - pad - 1), (190, 152, 84), max(2, th // 2), cv2.LINE_AA)
        m = np.zeros(out.shape[:2], bool); m[y0:y1] = True; out = outline(out, m, max(5, W // 90), (20, 16, 10))
        info = dict(belt_rows=[y0, y1], buckle_cols=cols.tolist())
    else: info = dict(belt_rows=None)
    return out, info
if __name__ == "__main__":
    for nm, fn in (("womanchest", chest), ("woman1", trousers)):
        o, info = fn(); Image.fromarray(o).save(D / f"{nm}_toon_512.png"); print(nm, info)
