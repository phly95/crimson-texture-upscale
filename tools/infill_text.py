"""infill_text.py: targeted lettering repair by klein inpainting with an OCR check.
infill_text(base, box, lines, ...) redraws only the pixels of a text box (SOURCE pixel coords x0,y0,x1,y1 scaled to the 4x picture) inside a 1024 px window of context, with the exact
wording in the prompt, tries several seeds, and keeps the draw whose OCR reading is closest to the wanted text (accepts at >= 0.97 similarity). Pasted back with a feathered mask, colour locked to the
surroundings. CLI: infill_text.py BASE4X OUT x0,y0,x1,y1 "line 1|line 2|..." ["extra style words"]   (several box/text pairs: repeat the group)"""
import base64, difflib, io, re, subprocess, sys, tempfile, os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tile_upscale as tu
import klein_server as ks

PRE = "Redraw the masked area so that it contains exactly this printed text, and nothing else: "
POST = " Same typeface, size, colour and spacing as the neighbouring print in this picture, crisp and sharp, every letter correctly formed and spelled exactly as given. Keep the paper texture; do not change anything outside the masked area."


def ocr(img):
    env = dict(os.environ, TESSDATA_PREFIX=str(Path.home() / ".local/share/tessdata"))
    best = ""
    for inv in (False, True):
        g = img.convert("L"); g = Image.eval(g, lambda v: 255 - v) if inv else g
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.png"; g.save(p)
            t = subprocess.run(["tesseract", str(p), "stdout", "--psm", "6"], capture_output=True, text=True, timeout=60, env=env).stdout
        t = " ".join(t.split())
        if len(t) > len(best):
            best = t
    return best


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def window(box, W, H, scale=4, T=1024):
    x0, y0, x1, y1 = [int(v * scale) for v in box]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    wx = min(max(0, cx - T // 2), W - T); wy = min(max(0, cy - T // 2), H - T)
    return (wx, wy, wx + T, wy + T), (x0 - wx, y0 - wy, x1 - wx, y1 - wy)


def infill(base, box, lines, extra="", tries=8, scale=4, seed0=1000, log=print, one_line=False):
    W, H = base.size
    win, (bx0, by0, bx1, by1) = window(box, W, H, scale)
    crop = base.crop(win).convert("RGB")
    T = crop.size[0]
    mask = np.zeros((T, T), np.uint8); mask[max(0, by0):by1, max(0, bx0):bx1] = 255
    want = " ".join(lines)
    if one_line:
        prompt = ("Redraw the masked strip so that it contains exactly one single line of small printed text, and nothing else: " + f'"{lines[0]}"' + ". Same typeface, size, colour and letter spacing as the neighbouring lines of print above and below it, "
                  "crisp, every letter correctly formed and spelled exactly as given, the line fills the strip from end to end. Keep the paper texture." + (" " + extra if extra else ""))
    else:
        prompt = PRE + " / ".join(f'"{l}"' for l in lines) + "." + POST + (" " + extra if extra else "")
    cb = io.BytesIO(); crop.save(cb, "PNG"); b64 = base64.b64encode(cb.getvalue()).decode()
    mb = io.BytesIO(); Image.fromarray(mask).save(mb, "PNG"); m64 = base64.b64encode(mb.getvalue()).decode()
    best = (-1, None)
    import fcntl
    for k in range(tries):
        with open("/tmp/klein_tile.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not ks.up():
                ks.start(Path(__file__).resolve().parents[1] / "pipeline" / "klein_server.log")
            r = ks._call("/sdcpp/v1/img_gen", {"prompt": prompt, "width": T, "height": T, "seed": seed0 + k, "output_format": "png", "init_image": b64, "mask_image": m64, "strength": 1.0}, timeout=30)
            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "o.png"; ks.collect(r["id"], out, timeout=600)
                res = Image.open(out).convert("RGB").resize((T, T), Image.LANCZOS)
        reg = res.crop((max(0, bx0), max(0, by0), bx1, by1))
        s = sim(ocr(reg), want)
        log(f"  seed {seed0 + k}: OCR similarity {s:.3f}")
        if s > best[0]:
            best = (s, res)
        if s >= 0.97:
            break
    s, res = best
    a = np.asarray(res, np.float32); c = np.asarray(crop, np.float32)
    m = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 6.0)[..., None]
    out = base.copy(); out.paste(Image.fromarray((c * (1 - m) + a * m).clip(0, 255).astype(np.uint8)), win[:2])
    return out, s


def line_bands(layer, box, scale=4, thr=0.5, min_row=4):
    """text lines inside a SOURCE box, found as rows of the text layer's alpha: list of (x0, y0, x1, y1) in source px"""
    a = np.asarray(layer.convert("RGBA").getchannel("A"), np.float32) / 255
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    sub = a[y0:y1, x0:x1] > thr
    rows = sub.sum(1) >= min_row
    bands, start = [], None
    for i, r in enumerate(list(rows) + [False]):
        if r and start is None:
            start = i
        elif not r and start is not None:
            bands.append([start, i]); start = None
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] < 2:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    out = []
    for b0, b1 in merged:
        cols = np.nonzero(sub[b0:b1].any(0))[0]
        out.append((x0 + cols.min() - 1, y0 + b0 - 1.5, x0 + cols.max() + 2, y0 + b1 + 1.5))
    return out


def infill_lines(base, layer, box, lines, extra="", log=print, **kw):
    bands = line_bands(layer, box)
    log(f"  {len(bands)} text lines found for {len(lines)} wanted lines")
    if len(bands) != len(lines):
        raise SystemExit(f"line count mismatch {len(bands)} vs {len(lines)}: {bands}")
    out, worst = base, 1.0
    for b, l in zip(bands, lines):
        out, sc = infill(out, b, [l], extra, one_line=True, log=log, **kw)
        log(f"  line '{l[:30]}': best OCR similarity {sc:.3f}"); worst = min(worst, sc)
    return out, worst


if __name__ == "__main__":
    if sys.argv[1] == "lines":                                     # lines BASE LAYER OUT x0,y0,x1,y1 "l1|l2|.."
        out, w = infill_lines(Image.open(sys.argv[2]), Image.open(sys.argv[3]), tuple(float(v) for v in sys.argv[5].split(",")), sys.argv[6].split("|"))
        out.save(sys.argv[4]); print("wrote", sys.argv[4], "worst line similarity", round(w, 3)); sys.exit(0)
    base = Image.open(sys.argv[1]); out = base
    args = sys.argv[3:]
    while args:
        box = tuple(float(v) for v in args[0].split(",")); lines = args[1].split("|")
        extra = ""
        if len(args) > 2 and not re.match(r"^[\d.]+,[\d.]+,[\d.]+,[\d.]+$", args[2]):
            extra = args[2]; args = args[3:]
        else:
            args = args[2:]
        out, s = infill(out, box, lines, extra); print("box", box, "best OCR similarity", round(s, 3))
    out.save(sys.argv[2]); print("wrote", sys.argv[2])
