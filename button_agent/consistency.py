"""consistency.py: cross-button consistency check of the pass-2 renderer + overview sheets.

For every rendered button (out/<name>.png if submitted, else sprites/<name>/draw.png) and each of the 4 states, measured on the
result reduced to the original size (what the game shows) with the same lettering detector the renderer uses on the originals:
  - median lettering colour (letter core pixels)
  - plate mean / std (Lab L) of the parchment panel away from the lettering
  - shadow depth and softness: mean luminance of the 1st..3rd row below the letters, relative to the plate
Outliers (> 3 robust sigmas from the median of all buttons, with a floor tolerance) are listed.
Writes out/_overview.png (all 59, 4 states, status-coloured captions) and out/_consistency.png (same letters from different buttons).

    venv/bin/python consistency.py
"""
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import button

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SPR = HERE / "sprites"
FONT = str(HERE / "fonts" / "BarlowCondensed-Bold.ttf")


def status():
    st = json.loads((HERE / "state.json").read_text()) if (HERE / "state.json").exists() else {}
    return {k: v.get("status", "todo") for k, v in st.items()}


def source(name):
    if (OUT / f"{name}.png").exists() and status().get(name) == "SUBMITTED":
        return OUT / f"{name}.png", "SUBMITTED"
    if (SPR / name / "draw.png").exists():
        return SPR / name / "draw.png", "rendered"
    return None, "missing"


def lab(rgb):
    return cv2.cvtColor(np.clip(rgb / 255, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)


def measure(name):
    spec = button.measure(name)
    p, _ = source(name)
    W, H, sh = spec["W"], spec["H"], spec["sh"]
    c = np.asarray(Image.open(p).convert("RGBA").resize((W, H), Image.BOX), np.float32)
    sts = [c[i * sh:(i + 1) * sh] for i in range(4)]
    tm = button.text_mask_1x(sts)
    if not tm.any():
        return None
    ys, xs = np.where(tm)
    y1 = ys.max() + 1
    res = []
    for s in range(4):
        dy = int(round(spec["rims"][s][0] - spec["rims"][2][0]))       # plates may step per state (EM_)
        S = np.roll(sts[s], -dy, 0)  # bring state s to state 3 alignment
        core = ndi.binary_erosion(tm) if ndi.binary_erosion(tm).sum() > 20 else tm
        letter = np.median(S[core][:, :3], 0)
        panel = np.zeros_like(tm)
        panel[6:sh - 10, 20:W - 20] = True
        panel &= ~ndi.binary_dilation(tm, iterations=4) & (S[..., 3] > 250)
        L = lab(S[..., :3])[..., 0]
        below = [L[y1 + k, xs.min():xs.max() + 1].mean() for k in (0, 1, 2) if y1 + k < sh]
        res.append(dict(letter=letter, plate_L=float(L[panel].mean()), plate_std=float(L[panel].std()),
                        plate_ab=lab(S[..., :3])[panel][:, 1:].mean(0),
                        shadow=float(L[panel].mean() - np.mean(below)), soft=float(below[-1] - below[0]) if len(below) == 3 else 0.))
    return res


def check(names):
    data = {n: measure(n) for n in names}
    data = {k: v for k, v in data.items() if v}
    print(f"{len(data)} lettered buttons measured")
    bad = []
    for s in range(4):
        print(f"\n--- state {s + 1} ---")
        for key, tol in (("letter", 6.0), ("plate_L", 2.5), ("plate_std", 1.5), ("shadow", 4.0), ("soft", 4.0)):
            vals = np.array([np.atleast_1d(data[n][s][key]) for n in data], np.float32)
            med = np.median(vals, 0); mad = np.median(np.abs(vals - med), 0) * 1.4826
            lim = np.maximum(3 * mad, tol)
            print(f"  {key:9s} median {np.round(med, 1).tolist()}  spread(1.48 MAD) {np.round(mad, 2).tolist()}")
            for n, v in zip(data, vals):
                if (np.abs(v - med) > lim).any():
                    bad.append((s + 1, key, n, np.round(v, 1).tolist(), np.round(med, 1).tolist()))
    print("\noutliers:" if bad else "\nno outliers")
    for b in bad:
        print("  state %d %-9s %-26s %s (median %s)" % b)
    return bad


def overview(names):
    st = status()
    cols, tw = 4, 480
    cells = []
    for n in names:
        p, kind = source(n)
        if p is None:
            im = Image.open(SPR / n / "orig.png").convert("RGBA"); kind = "BLOCKED (original shown)"
        else:
            im = Image.open(p).convert("RGBA")
        th = int(round(im.height * tw / im.width))
        cells.append((n, im.resize((tw, th), Image.LANCZOS), st.get(n, "todo") if kind == "rendered" else kind))
    rowh = []
    for i in range(0, len(cells), cols):
        rowh.append(max(c[1].height for c in cells[i:i + cols]) + 26)
    pad = 10
    canvas = Image.new("RGB", (cols * (tw + pad) + pad, sum(rowh) + pad), (34, 36, 40))
    d = ImageDraw.Draw(canvas)
    f = ImageFont.truetype(FONT, 18)
    y = pad
    for r, i in enumerate(range(0, len(cells), cols)):
        for j, (n, im, kind) in enumerate(cells[i:i + cols]):
            x = pad + j * (tw + pad)
            canvas.paste(im, (x, y), im)
            colr = (120, 220, 120) if kind == "SUBMITTED" else (230, 190, 90) if kind in ("PASSED", "attempted", "rendered") else (230, 110, 100)
            d.text((x + 2, y + im.height + 3), f"{n}  [{kind}]", fill=colr, font=f)
        y += rowh[r]
    canvas.save(OUT / "_overview.png")
    print("wrote", OUT / "_overview.png", canvas.size)


def letters_sheet():
    """the same letters taken from different buttons (state 2), side by side"""
    picks = [("A", ["PF_B_AUDIO", "MM_B_CAMPAIGN", "PX_B_SELLPLANES", "CM_B_CANCEL"]),
             ("O", ["PF_B_CONTROLS", "PF_B_VIDEO", "CR_B_ABOUT", "LS_B_LOAD"]),
             ("E", ["EM_B_RESUME", "PF_B_MAINMENU", "MS_B_DONE", "CM_B_CANCEL"])]
    cw, chh = 260, 120
    canvas = Image.new("RGB", (100 + 4 * (cw + 8), len(picks) * (chh + 10) + 10), (34, 36, 40))
    d = ImageDraw.Draw(canvas); f = ImageFont.truetype(FONT, 26)
    for r, (ch, names) in enumerate(picks):
        d.text((20, 10 + r * (chh + 10) + chh // 2 - 14), ch, fill=(230, 230, 230), font=f)
        for j, n in enumerate(names):
            p, _ = source(n)
            if p is None:
                continue
            im = Image.open(p).convert("RGBA"); spec = button.measure(n); sh = spec["sh"] * 5; t = spec["text"]
            cx = (t["x0"] + t["x1"]) * 5 // 2; cy = sh + (t["y0"] + t["y1"]) * 5 // 2
            crop = im.crop((cx - cw // 2, cy - chh // 2, cx + cw // 2, cy + chh // 2))
            canvas.paste(crop, (100 + j * (cw + 8), 10 + r * (chh + 10)), crop)
    canvas.save(OUT / "_consistency.png")
    print("wrote", OUT / "_consistency.png")


if __name__ == "__main__":
    order = json.loads((HERE / "sprites.json").read_text())["order"]
    lettered = [n for n in order if n not in button.BLANK and source(n)[0] is not None]
    check(lettered)
    overview(order)
    letters_sheet()
