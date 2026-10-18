"""textlayer.py: lettering as a separate layer, so FLUX only ever redraws lettering-free material.

  split(src, work)  -> work/plate.png (the picture with all lettering/logos/stamps removed, at the source size), work/layer.png (RGBA text layer at the source size)
  restore(plate4x, layer, out, scale)  -> composite the layer back over the upscaled plate with crisp edges
  check(img)        -> number of confident OCR words left in a picture

split(): klein is asked (edit mode, whole picture in one 1024 px pass) to remove every piece of lettering and fill the gap with the surrounding texture. Only the pixels that really
changed in the text sense (high-pass difference between the original and the klein result) are taken from klein; everything else stays the ORIGINAL picture, so the paper/wood is not
redrawn at this stage. If OCR still finds words in the plate, klein is asked again with another seed (up to TRIES), and leftovers are added to the mask.
The layer's alpha is the per-pixel ink fraction (projection of orig-plate on the stroke colour), its colour the original's own, bled outward.
restore(): alpha is upscaled (Lanczos) and its edge re-sharpened (contrast around 0.5), colour upscaled with bicubic, composited over the plate.
"""
import os, subprocess, sys, tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "tools" / "pipeline"))
import tile_upscale as tu                                                        # noqa: E402

PROMPT = ("Remove every piece of text, lettering, numbers, logos, emblems, stamps, signatures and handwriting from this picture. Where they were, continue the surrounding "
          "paper, cloth, metal or background texture seamlessly, as if nothing had ever been printed there. Do not change anything else: keep every other object, border, line, "
          "shadow, fold and colour exactly as it is. {extra}")
TRIES = 3


def ocr_words(img, min_conf=55):
    env = dict(os.environ, TESSDATA_PREFIX=str(Path.home() / ".local/share/tessdata"))
    g = img.convert("L"); g = g.resize((g.size[0] * 2, g.size[1] * 2), Image.LANCZOS)
    out = []
    for inv in (False, True):
        im2 = Image.eval(g, lambda v: 255 - v) if inv else g
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.png"; im2.save(p)
            r = subprocess.run(["tesseract", str(p), "stdout", "--psm", "11", "tsv"], capture_output=True, text=True, timeout=120, env=env).stdout
        for ln in r.splitlines()[1:]:
            f = ln.split("\t")
            if len(f) >= 12 and f[11].strip():
                try:
                    c = float(f[10])
                except ValueError:
                    continue
                w = f[11].strip()
                if c >= min_conf and sum(ch.isalnum() for ch in w) >= 3:
                    out.append((w, int(f[6]) // 2, int(f[7]) // 2, int(f[8]) // 2, int(f[9]) // 2))
    return out


def klein_clean(orig, seed, extra=""):
    W, H = orig.size
    s = 1024 / max(W, H)
    mw, mh = int(round(W * s / 16)) * 16, int(round(H * s / 16)) * 16
    with tempfile.TemporaryDirectory() as td:
        ref, out = Path(td) / "r.png", Path(td) / "o.png"
        orig.resize((mw, mh), Image.LANCZOS).save(ref)
        tu.locked_gen_wh(ref, out, PROMPT.format(extra=extra).strip(), mw, mh, seed)
        return Image.open(out).convert("RGB").resize((W, H), Image.LANCZOS)


def text_mask(orig, plate, thr=0.10):
    """pixels where lettering was removed: strong difference of the high-passed lightness (grain differences are weak, ink is strong)"""
    o = np.asarray(orig.convert("L"), np.float32) / 255; p = np.asarray(plate.convert("L"), np.float32) / 255
    ho, hp = o - cv2.GaussianBlur(o, (0, 0), 3.0), p - cv2.GaussianBlur(p, (0, 0), 3.0)
    d = np.abs(ho - hp)
    d = np.maximum(d, np.abs(cv2.GaussianBlur(o - p, (0, 0), 1.0)) * 0.6)         # filled blobs (solid logo areas) show in the plain difference too
    m = (d > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 6
    m = keep[lab].astype(np.uint8)
    return cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=2)


def thin_structures(orig, near, t=0.11, reach=10):
    """faint remnants of lettering: thin dark or light strokes (black/top-hat of the lightness) within `reach` px of what klein already removed"""
    g = np.asarray(orig.convert("L"), np.float32) / 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = np.maximum(cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k), cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)) > t
    hull = cv2.dilate(near, np.ones((2 * reach + 1, 2 * reach + 1), np.uint8)) > 0
    return (th & hull).astype(np.uint8)


def inpaint(orig, m):
    bgr = cv2.cvtColor(np.asarray(orig), cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, (m * 255).astype(np.uint8), 4, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def split(src, work, seed=1, extra=""):
    """klein finds WHERE the lettering is (difference between the original and its 'remove all text' edit); the plate itself is the original with those pixels (plus faint
    remnants next to them) inpainted from their surroundings: smooth, no redrawn content. FLUX re-textures the plate later."""
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    orig = Image.open(src).convert("RGB"); W, H = orig.size
    best = None
    for k in range(TRIES):
        kl = klein_clean(orig, seed + k, extra)
        kl.save(work / f"klein_clean_{k}.png")
        lo = tu.colour_lock(np.asarray(kl).astype(np.float32) / 255, np.asarray(orig).astype(np.float32) / 255, 10.0, 24.0)
        kl = Image.fromarray((lo * 255 + .5).astype(np.uint8))
        m = text_mask(orig, kl)
        m = cv2.dilate(np.maximum(m, thin_structures(orig, m)), np.ones((3, 3), np.uint8))
        plate = inpaint(orig, m)
        words = ocr_words(plate)
        print(f"  clean try {k}: mask {100 * m.mean():.1f}% of the picture, {len(words)} words left", flush=True)
        if best is None or len(words) < best[0]:
            best = (len(words), plate, m, words)
        if len(words) == 0:
            break
    n, plate, m, words = best
    if words:                                                                    # leftovers: add their boxes (stroke pixels only) and inpaint again
        m = m.copy()
        for _, x, y, ww, hh in words:
            box = np.zeros_like(m); box[max(0, y - 3):y + hh + 3, max(0, x - 3):x + ww + 3] = 1
            m = np.maximum(m, thin_structures(orig, box, reach=0) & box)
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        plate = inpaint(orig, m)
    plate.save(work / "plate.png")
    make_layer(orig, plate, m).save(work / "layer.png")
    Image.fromarray((m * 255).astype(np.uint8)).save(work / "mask.png")
    return {"mask_pct": round(100 * float(m.mean()), 2), "words_left_in_plate": len(ocr_words(plate))}


def make_layer(orig, plate, m):
    """RGBA text layer at the source size. alpha = how much of the pixel is ink: ||orig-plate|| relative to the strongest difference nearby (works for multi-colour logos too);
    colour = the ink colour unmixed from the background, so that  plate*(1-a) + colour*a  gives back the original exactly at 1x"""
    o, p = np.asarray(orig).astype(np.float32), np.asarray(plate).astype(np.float32)
    dn = np.linalg.norm(o - p, axis=-1) * (m > 0)
    near = cv2.dilate(dn, np.ones((13, 13), np.uint8))
    alpha = np.clip(dn / np.maximum(0.85 * near, 24.0), 0, 1)
    alpha[dn < 8] = 0
    col = np.clip(p + (o - p) / np.maximum(alpha, 0.3)[..., None], 0, 255)
    col = np.where((alpha > 0)[..., None], col, o)
    return Image.fromarray(np.dstack([col, alpha * 255]).astype(np.uint8), "RGBA")


def restore(plate4x, layer, out, scale=4, sharp=2.4):
    pl = Image.open(plate4x).convert("RGB"); W, H = pl.size
    ly = layer if isinstance(layer, Image.Image) else Image.open(layer)
    a = np.asarray(ly.getchannel("A").resize((W, H), Image.LANCZOS), np.float32) / 255
    a = np.clip((a - 0.5) * sharp + 0.5, 0, 1)
    c = np.asarray(ly.convert("RGB").resize((W, H), Image.BICUBIC), np.float32)
    res = np.asarray(pl, np.float32) * (1 - a[..., None]) + c * a[..., None]
    Image.fromarray(res.clip(0, 255).astype(np.uint8)).save(out)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "split":
        print(split(sys.argv[2], sys.argv[3]))
    elif cmd == "restore":
        restore(sys.argv[2], sys.argv[3], sys.argv[4]); print("ok")
