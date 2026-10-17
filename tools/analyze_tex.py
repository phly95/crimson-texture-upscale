"""analyze_tex.py IMAGE [--json OUT.json] [--overlay OUT.png]

Deterministic measurements of a tiny game texture, written as a compact text digest that the prompt-writing VLM reads
alongside the picture.  Coordinates are percent of the frame, x to the right, y downward, origin top-left.  Nothing
here is random (k-means is seeded) and nothing touches the game install.

    from analyze_tex import analyze
    text, data = analyze("path.png")
"""
import json, math, subprocess, sys, tempfile, warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color as skc, feature, measure

warnings.filterwarnings("ignore")

NAMED = {  # coarse colour names, matched in Lab
    "black": (10, 10, 10), "near-black": (35, 35, 38), "dark grey": (70, 70, 72), "grey": (125, 125, 125), "light grey": (180, 180, 180),
    "white": (245, 245, 245), "cream": (238, 228, 200), "tan": (196, 170, 120), "beige": (215, 195, 160), "brown": (110, 70, 40),
    "dark brown": (60, 40, 28), "rust": (150, 70, 35), "red": (200, 30, 30), "dark red": (120, 20, 30), "crimson": (160, 25, 45),
    "orange": (235, 120, 25), "gold": (205, 160, 30), "yellow": (240, 210, 40), "olive": (110, 110, 50), "dark olive green": (55, 70, 40),
    "dark green": (35, 80, 50), "mid green": (70, 130, 65), "light green": (150, 200, 110), "sage green": (150, 165, 130), "teal": (30, 110, 110),
    "dark teal": (30, 75, 80), "sky blue": (130, 180, 225), "blue": (50, 90, 190), "denim blue": (65, 100, 150), "navy": (25, 35, 80),
    "purple": (110, 60, 150), "pink": (230, 140, 170), "steel": (110, 125, 140), "bright green": (0, 255, 0),
}
_NL = skc.rgb2lab(np.array([list(v) for v in NAMED.values()], dtype=np.float64)[None] / 255)[0]
_NN = list(NAMED)


def cname(rgb):
    lab = skc.rgb2lab(np.array(rgb, dtype=np.float64)[None, None] / 255)[0, 0]
    return _NN[int(np.argmin(((_NL - lab) ** 2).sum(1)))]


def pct(v, n):
    return int(round(100 * v / n))


def dirword(theta_deg):
    """line direction in degrees (0 = horizontal, 90 = vertical, image y down) -> word"""
    t = theta_deg % 180
    if t < 15 or t >= 165: return "horizontal"
    if 75 <= t < 105: return "vertical"
    return "diagonal (down-right)" if t < 90 else "diagonal (up-right)"


ALPHA = None
SOFT_ONLY = None


def load(path):
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im).astype(np.float64)
    global ALPHA, SOFT_ONLY
    ALPHA = a[..., 3]
    m = a[..., 3] >= 128
    SOFT_ONLY = None
    if m.sum() < 20 and (a[..., 3] >= 8).sum() >= 20:      # a translucent decal (shadow, glow): nothing reaches alpha 128
        SOFT_ONLY = int(a[..., 3].max())
        m = a[..., 3] >= 8
    return a[..., :3], m


def kmeans_clusters(rgb, mask, k=8):
    lab = skc.rgb2lab(rgb / 255)
    pts = lab[mask].astype(np.float32)
    if len(pts) < 20: return np.full(mask.shape, -1), []
    cv2.setRNGSeed(0)
    k = min(k, max(2, len(pts) // 30))
    _, lbl, cen = cv2.kmeans(pts, k, None, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 60, 0.05), 4, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.ravel(); cen = cen.copy()
    # merge clusters closer than dE 9
    alive = list(range(k)); merged = True
    while merged:
        merged = False
        for i in alive:
            for j in alive:
                if i < j and np.linalg.norm(cen[i] - cen[j]) < 9:
                    ni, nj = (lbl == i).sum(), (lbl == j).sum()
                    cen[i] = (cen[i] * ni + cen[j] * nj) / (ni + nj); lbl[lbl == j] = i; alive.remove(j); merged = True; break
            if merged: break
    order = sorted(alive, key=lambda c: -(lbl == c).sum())
    remap = {c: n for n, c in enumerate(order)}
    full = np.full(mask.shape, -1, dtype=int)
    full[mask] = np.array([remap[v] for v in lbl])
    return full, [cen[c] for c in order]


def ocr_boxes(gray8, W, H, min_conf=70):
    """confident English words with boxes in native pixels: [(text, conf, x0, y0, x1, y1)]; empty when tesseract or its model is missing"""
    env = dict(__import__("os").environ, TESSDATA_PREFIX=str(Path.home() / ".local/share/tessdata"))
    big = Image.fromarray(gray8).resize((W * 4, H * 4), Image.BICUBIC)
    best = {}
    try:
        for inv in (False, True):
            im2 = Image.eval(big, lambda v: 255 - v) if inv else big
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "t.png"; im2.save(p)
                r = subprocess.run(["tesseract", str(p), "stdout", "--psm", "11", "tsv"], capture_output=True, text=True, timeout=40, env=env).stdout
            for ln in r.splitlines()[1:]:
                f = ln.split("\t")
                if len(f) >= 12:
                    w = f[11].strip()
                    try:
                        c = float(f[10])
                    except ValueError:
                        continue
                    if c >= min_conf and len(w) >= 3 and w.replace("'", "").replace(".", "").isalpha():
                        x0, y0, bw, bh = (int(f[6]) / 4, int(f[7]) / 4, int(f[8]) / 4, int(f[9]) / 4)
                        if w not in best or best[w][0] < c:
                            best[w] = (c, x0, y0, x0 + bw, y0 + bh)
    except Exception:                                           # noqa: BLE001
        return []
    return [(w, *v) for w, v in best.items()]


def stud_rows(lum, mask):
    """rows/columns of small evenly spaced dots (rivets, nails, studs, bolts); two sensitivities, the sensitive one needs stricter regularity"""
    H, W = lum.shape
    f = lum.astype(np.float32)
    dog = cv2.GaussianBlur(f, (0, 0), 0.8) - cv2.GaussianBlur(f, (0, 0), 2.2)
    big = cv2.GaussianBlur(f, (0, 0), 2.5) - cv2.GaussianBlur(f, (0, 0), 6.0)      # response at 3x the dot scale: a real small dot has little of it
    sd = float(np.median(np.abs(dog[mask])) / 0.6745) if mask.any() else 0.0     # robust noise scale, not dominated by big edges
    gx_ = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3); gy_ = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    box = lambda a: cv2.blur(a, (5, 5))                          # noqa: E731
    jxx, jyy, jxy = box(gx_ * gx_), box(gy_ * gy_), box(gx_ * gy_)
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)   # ~0 for a round dot, ~1 for a straight edge
    found = []
    for k_sd, reg_max, n_min, cap in ((3.5, 0.45, 4, 0.06), (2.0, 0.30, 4, 0.10)):
        for pol in (1, -1):
            d = pol * dog
            mx = (d == ndi.maximum_filter(d, size=3)) & (d > max(2.5, k_sd * sd)) & mask & (coh < 0.85)
            ys, xs = np.nonzero(mx)
            pts = list(zip(xs.tolist(), ys.tolist()))
            if len(pts) > cap * H * W or len(pts) < n_min:
                continue
            for axis in (1, 0):                                 # axis 1: group by y (a horizontal row); 0: group by x (a column)
                tol = max(1.4, 0.02 * (H if axis == 1 else W))
                pts_s = sorted(pts, key=lambda p: p[axis]); i = 0
                while i < len(pts_s):
                    grp = [pts_s[i]]; j = i + 1
                    while j < len(pts_s) and pts_s[j][axis] - grp[0][axis] <= tol:
                        grp.append(pts_s[j]); j += 1
                    if len(grp) >= n_min:
                        along = sorted(p[1 - axis] for p in grp)
                        diffs = np.diff(along); diffs = diffs[diffs > 0.8]
                        L = W if axis == 1 else H
                        if len(diffs) >= 3:
                            med = float(np.median(diffs))
                            if med >= max(2.5, 0.035 * L) and len(grp) <= 20 and diffs.max() <= 2.3 * med and float(np.std(diffs)) / med < reg_max:
                                # false-positive gates (windows, tile/checker patterns, grain, markings also form regular rows): the row must be
                                # very regular, stand out against few other peaks, be made of isolated small dots (not parts of bigger shapes)
                                resp = np.array([d[y, x] for (x, y) in grp]); br = np.array([abs(pol * big[y, x]) for (x, y) in grp])
                                iso = float(np.median(br / (resp + 1e-6)))
                                gated = float(np.std(diffs)) / med > 0.20 or len(pts) > 45 or iso > 0.6
                                sizes = []
                                for (x, y) in grp:                                  # dot diameter: pixels above half the local contrast
                                    sl = d[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
                                    sizes.append(float((sl > 0.5 * sl.max()).sum()) ** 0.5)
                                pos = float(np.mean([p[axis] for p in grp]))
                                if not gated and float(np.median(sizes)) <= 0.5 * med:       # a nail head is small next to its spacing
                                  found.append((len(grp) * med * (2 if k_sd == 3.5 else 1), len(grp), axis, pos, (along[0], along[-1]), med, "bright" if pol > 0 else "dark", float(np.median(sizes))))
                    i = j if j > i + 1 else i + 1
    found.sort(reverse=True)
    if found:                                                   # one surface, one kind of nail: keep the polarity with more dots
        tot = {"bright": 0, "dark": 0}
        for r_ in found: tot[r_[6]] += r_[1]
        keep = max(tot, key=tot.get)
        found = [r_ for r_ in found if r_[6] == keep]
    out, used, allpts = [], [], []
    for _, n, axis, pos, span, med, kind, dia in found:
        L = H if axis == 1 else W
        key = (axis, round(pos / L * 18))
        if key in used or (axis, key[1] - 1) in used or (axis, key[1] + 1) in used: continue
        used.append(key)
        size_txt = f", each dot about {dia:.1f}px in the original (~{100*dia/W:.0f}% of the width)"
        if axis == 1:
            out.append(f"a horizontal row of {n} small {kind} dots at y {pct(pos,H)}%, from x {pct(span[0],W)}% to {pct(span[1],W)}%, spaced about {100*med/W:.0f}% apart{size_txt}")
        else:
            out.append(f"a vertical column of {n} small {kind} dots at x {pct(pos,W)}%, from y {pct(span[0],H)}% to {pct(span[1],H)}%, spaced about {100*med/H:.0f}% apart{size_txt}")
        if len(out) >= 5: break
    if out:
        out.append("IF you judge these to be nails/rivets (the surface is a metal, wood or plate surface and small round heads are plausible at those places), they are all the same kind of head (same colour, same size, evenly spaced), a clearly visible round metal head with a small highlight and a small shadow once enlarged; never multicoloured and never random pebbles. If the dots are part of something else in the picture (windows, pupils, tiles, markings, lamps, ground detail), do NOT add nails")
    return out


def analyze(path, overlay=None):
    rgb, mask = load(path)
    H, W = mask.shape
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    n_op = int(mask.sum()); out = []; data = {"size": [W, H]}
    out.append(f"FRAME {W}x{H}px. Coordinates below are percent of the frame (x right, y down, origin top-left).")
    if SOFT_ONLY is not None:
        out.append(f"ALL drawn pixels are only partly transparent (the most opaque pixel is alpha {SOFT_ONLY}/255): this is a soft translucent decal such as a shadow, glow or stain, not a solid picture; the masks below use alpha >= 8.")

    # ---- silhouette
    tr = 1 - n_op / (H * W)
    data["empty_frac"] = tr
    if tr > 0.02:
        ys, xs = np.nonzero(mask)
        try:
            hull = measure.regionprops(mask.astype(np.uint8))[0]
        except IndexError:
            hull = type('H', (), {'solidity': 0.0})()
        lab_cc, ncc = ndi.label(mask)
        out.append(f"SILHOUETTE (cut-out, {tr*100:.0f}% of the frame is empty): bounding box x {pct(xs.min(),W)}-{pct(xs.max()+1,W)}, y {pct(ys.min(),H)}-{pct(ys.max()+1,H)}; "
                   f"centre of mass ({pct(xs.mean(),W)},{pct(ys.mean(),H)}); {ncc} separate piece(s); solidity {hull.solidity:.2f} (1 = blob, low = spiky/sprawling).")
        # border profile by distance inward from the outermost visible pixel (alpha >= 16 so the soft white halo counts)
        anyv = ALPHA >= 16
        d = ndi.distance_transform_edt(anyv)
        semi = ((ALPHA > 0) & (ALPHA < 250)).mean()
        prof = []
        for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8)]:
            m = anyv & (d > lo) & (d <= hi)
            if m.sum() > 10:
                px = rgb[m]; L = px.mean(1)
                prof.append((lo, hi, tuple(int(v) for v in px.mean(0)), float(ALPHA[m].mean()), float((L > 170).mean()), float((L < 60).mean())))
        if prof:
            txt = "; ".join(f"{lo}-{hi}px in: {cname(c)} RGB{c}, mean alpha {a:.0f}/255" for lo, hi, c, a, fl, fd in prof)
            summ = []
            if prof[0][4] > 0.6: summ.append(f"the outermost ring is a light (white/cream) halo, about {prof[0][1]}px ({100*prof[0][1]/W:.1f}% of the width) and partly see-through")
            for lo, hi, c, a, fl, fd in prof[1:3]:
                if fd > 0.45: summ.append(f"a dark rim lies inside it at {lo}-{hi}px (an ink outline only if the picture itself shows ink lines; otherwise just an anti-aliasing fringe)"); break
            out.append("OUTLINE/BORDER (distance inward from the cut-out edge): " + txt + ". " + ("Reading: " + "; ".join(summ) + "." if summ else "No distinct border ring.")
                       + f" {semi*100:.0f}% of pixels are partly transparent (soft alpha rim).")

    # ---- alpha character: holes, translucency, edge softness
    if tr > 0.02 or (ALPHA < 250).mean() > 0.02:
        a = ALPHA
        op = (a >= 240).mean(); par = ((a >= 16) & (a < 240)).mean(); emp = (a < 16).mean()
        bnd = mask ^ ndi.binary_erosion(mask)
        ramp = ((a >= 16) & (a < 240)).sum() / max(1, bnd.sum())
        notes = [f"{op*100:.0f}% fully opaque, {par*100:.0f}% partly transparent (alpha 16-240), {emp*100:.0f}% empty"]
        if par > 0.25: notes.append("the subject itself is translucent/soft (smoke, glow, haze, glass, fringe): keep it soft and see-through, with NO hard outline")
        notes.append(f"the edge ramps from empty to solid over about {ramp:.1f}px (~{100*ramp/W:.1f}% of the width): " + ("a soft/feathered edge, keep it soft" if ramp >= 2.0 else "a crisp edge"))
        out.append("ALPHA CHARACTER: " + "; ".join(notes) + ".")
        lab_e, ne = ndi.label(a < 128)
        holes = []
        for k in range(1, ne + 1):
            ys, xs = np.nonzero(lab_e == k)
            if xs.min() == 0 or ys.min() == 0 or xs.max() == W - 1 or ys.max() == H - 1: continue
            if len(xs) < 0.002 * W * H: continue
            if a[lab_e == k].mean() > 45 or (a[lab_e == k] < 16).mean() < 0.5: continue          # a ragged, half-transparent gap (foliage, fur, cloth edge) is not a window: only genuinely empty regions are holes
            bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
            fill = len(xs) / (bw * bh)
            shape = "round/oval" if abs(fill - 0.785) < 0.1 else ("rectangular" if fill > 0.88 else "irregular")
            holes.append((len(xs), f"{shape} hole x {pct(xs.min(),W)}-{pct(xs.max()+1,W)}, y {pct(ys.min(),H)}-{pct(ys.max()+1,H)}"))
        if 0.003 < par <= 0.25:
            lab_t, nt = ndi.label((a >= 16) & (a < 240) & ndi.binary_erosion(a >= 16, iterations=2))
            tp = []
            for k in range(1, nt + 1):
                ys, xs = np.nonzero(lab_t == k)
                if len(xs) >= 0.003 * W * H:
                    pc = tuple(int(v) for v in rgb[lab_t == k].mean(0))
                    tp.append((len(xs), f"x {pct(xs.min(),W)}-{pct(xs.max()+1,W)}, y {pct(ys.min(),H)}-{pct(ys.max()+1,H)} (colour {cname(pc)} RGB{pc}, mean alpha {a[lab_t==k].mean():.0f}/255)"))
            if tp:
                tp.sort(reverse=True)
                out.append("TRANSLUCENT PATCHES inside the shape (glass, windows, thin cloth; the transparency is restored later, so paint each patch OPAQUE in its own measured colour as tinted glass or cloth, never green): " + "; ".join(t for _, t in tp[:6]) + ".")
        if holes:
            holes.sort(reverse=True)
            out.append(f"CUT-OUT HOLES inside the shape (windows, gaps; the redraw must keep them empty, on the green background): {len(holes)}: " + "; ".join(h for _, h in holes[:8]) + ".")
        else:
            out.append("CUT-OUT HOLES: none inside the shape.")

    # ---- colour clusters and layout map
    lbl, cents = kmeans_clusters(rgb, mask)
    letters = "ABCDEFGHIJ"
    data["clusters"] = []
    rows = []
    for ci, c in enumerate(cents):
        m = lbl == ci
        area = m.sum() / (H * W)
        col = tuple(int(v) for v in (skc.lab2rgb(c[None, None].astype(np.float64))[0, 0] * 255).round())
        ys, xs = np.nonzero(m)
        cc, n = ndi.label(m, structure=np.ones((3, 3)))
        sizes = ndi.sum(m, cc, range(1, n + 1)) if n else []
        big = [i + 1 for i in np.argsort(sizes)[::-1][:3] if sizes[i] >= 0.004 * H * W]
        parts = []
        for b in big:
            yy, xx = np.nonzero(cc == b)
            parts.append(f"piece x {pct(xx.min(),W)}-{pct(xx.max()+1,W)} y {pct(yy.min(),H)}-{pct(yy.max()+1,H)} ({100*len(yy)/(H*W):.1f}%)")
        touch = []
        if xs.min() == 0: touch.append("left")
        if xs.max() == W - 1: touch.append("right")
        if ys.min() == 0: touch.append("top")
        if ys.max() == H - 1: touch.append("bottom")
        rows.append(f"  {letters[ci]} = {cname(col)} RGB{col}, {area*100:.1f}% of frame, centre ({pct(xs.mean(),W)},{pct(ys.mean(),H)}), spread x{pct(xs.std(),W)} y{pct(ys.std(),H)}"
                    + (f", touches {'/'.join(touch)} edge" if touch else "") + (": " + "; ".join(parts) if parts else ""))
        data["clusters"].append({"id": letters[ci], "name": cname(col), "rgb": col, "area": float(area)})
    if rows:
        out.append("COLOUR REGIONS (k-means in Lab, biggest first; letters are used in the layout map):\n" + "\n".join(rows))
        G = 16
        lines = []
        for gy in range(G):
            s = ""
            for gx in range(G):
                blk = lbl[gy * H // G:(gy + 1) * H // G, gx * W // G:(gx + 1) * W // G]
                v = blk[blk >= 0]
                s += "." if len(v) < 0.5 * blk.size else letters[np.bincount(v).argmax()]
            lines.append(s)
        out.append("LAYOUT MAP, 16x16 cells (top row first, each cell = most common colour region; '.' = empty):\n" + "\n".join("  " + l for l in lines))

    # ---- heatmaps (digits 0-9, 12x12)
    g = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 0.7)
    gx_ = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy_ = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx_, gy_); mag[~mask] = 0

    def digitmap(arr, valid, G=12, lo=None, hi=None):
        cells = np.full((G, G), np.nan)
        for y in range(G):
            for x in range(G):
                sl = (slice(y * H // G, (y + 1) * H // G), slice(x * W // G, (x + 1) * W // G))
                v = valid[sl]
                if v.mean() > 0.5: cells[y, x] = arr[sl][v].mean()
        f = cells[~np.isnan(cells)]
        if len(f) == 0: return ["  " + "." * G] * G
        lo = np.percentile(f, 2) if lo is None else lo; hi = np.percentile(f, 98) if hi is None else hi
        txt = []
        for y in range(G):
            txt.append("  " + "".join("." if np.isnan(cells[y, x]) else str(int(np.clip((cells[y, x] - lo) / (hi - lo + 1e-9) * 9.99, 0, 9))) for x in range(G)))
        return txt
    out.append("BRIGHTNESS MAP 12x12 (0 darkest .. 9 brightest, relative to this picture; '.' = empty):\n" + "\n".join(digitmap(lum, mask)))
    out.append("DETAIL/EDGE MAP 12x12 (0 flat .. 9 many strong edges, relative to this picture):\n" + "\n".join(digitmap(mag, mask)))

    # ---- orientation + periodicity per 3x3 region
    names = [["top-left", "top-centre", "top-right"], ["middle-left", "centre", "middle-right"], ["bottom-left", "bottom-centre", "bottom-right"]]
    jxx, jyy, jxy = gx_ * gx_, gy_ * gy_, gx_ * gy_
    reg = []
    energy_all = (jxx + jyy)[mask].mean() + 1e-9

    def struct(sl):
        v = mask[sl]
        if v.sum() < 30: return None
        a, b, c = jxx[sl][v].mean(), jyy[sl][v].mean(), jxy[sl][v].mean()
        e = a + b
        coh = math.hypot(a - b, 2 * c) / (e + 1e-9)
        th = 0.5 * math.degrees(math.atan2(2 * c, a - b))   # gradient direction
        return e, coh, (th + 90) % 180                       # line direction

    def period(sl):
        tile = lum[sl].astype(np.float32)
        if tile.shape[0] < 12 or tile.shape[1] < 12: return None
        tile = tile - tile.mean()
        win = np.outer(np.hanning(tile.shape[0]), np.hanning(tile.shape[1]))
        F = np.abs(np.fft.fftshift(np.fft.fft2(tile * win))) ** 2
        cy, cx = np.array(F.shape) // 2
        yy, xx = np.indices(F.shape)
        fy, fx = (yy - cy) / F.shape[0], (xx - cx) / F.shape[1]
        r = np.hypot(fx, fy)
        ok = (r > 4 / min(tile.shape)) & (r < 1 / 2.5)
        if ok.sum() < 10: return None
        Fm = np.where(ok, F, 0)
        i = np.unravel_index(Fm.argmax(), F.shape)
        ratio = F[i] / (F[ok].mean() + 1e-9)
        if ratio < 20: return None
        per = 1 / np.hypot(fx[i], fy[i])
        line_dir = (math.degrees(math.atan2(fy[i], fx[i])) + 90) % 180
        return per, line_dir, ratio

    for gy in range(3):
        for gx in range(3):
            sl = (slice(gy * H // 3, (gy + 1) * H // 3), slice(gx * W // 3, (gx + 1) * W // 3))
            s = struct(sl)
            if s is None: continue
            e, coh, th = s
            txt = f"  {names[gy][gx]}: "
            if e < 0.12 * energy_all: txt += "very smooth, almost no texture"
            elif coh > 0.30: txt += f"streaks/grain/rows run {dirword(th)} (coherence {coh:.2f})"
            else: txt += "isotropic mottling, no dominant direction"
            p = period(sl)
            if p: txt += f"; regular repeat about every {p[0]:.1f}px (~{100*p[0]/W:.1f}% of width) in lines running {dirword(p[1])}"
            reg.append(txt)
    if energy_all > 15000:
        out.append("TEXTURE DIRECTION: skipped, this is strongly outlined graphic art (edge energy %d); stripe/grain direction is not meaningful here." % energy_all)
    elif reg:
        out.append("TEXTURE DIRECTION per region (structure tensor + FFT; 'coherence' 0 = none, 1 = perfectly straight lines):\n" + "\n".join(reg))
    # mottle scale
    hp = lum - cv2.GaussianBlur(lum.astype(np.float32), (0, 0), max(W, H) / 12)
    F = np.abs(np.fft.fft2(hp * mask)) ** 2
    fy_, fx_ = np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None]
    r = np.hypot(fx_, fy_); band = (r > 1 / (max(W, H) / 2.5)) & (r < 0.45)
    if band.sum() and F[band].sum() > 0 and energy_all <= 15000:
        fc = (r[band] * F[band]).sum() / F[band].sum()
        out.append(f"GRAIN/MOTTLE SCALE: typical detail blob size is about {1/fc:.1f}px (~{100/fc/W:.1f}% of the width); overall edge energy {energy_all:.0f} (flat art is low, busy photo is high).")
        data["blob_px"] = 1 / fc

    # ---- straight lines
    gray8 = np.clip(lum, 0, 255).astype(np.uint8)
    work = cv2.resize(gray8, (W * 4, H * 4), interpolation=cv2.INTER_CUBIC) if max(W, H) <= 128 else gray8
    S = work.shape[1] / W
    ed = cv2.Canny(cv2.GaussianBlur(work, (0, 0), max(1.5, 0.012 * work.shape[1])), 20, 60)
    lines = cv2.HoughLinesP(ed, 1, np.pi / 180, int(0.18 * max(W, H) * S), minLineLength=int(0.28 * max(W, H) * S), maxLineGap=int(0.06 * max(W, H) * S))
    found = []
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            L = math.hypot(x2 - x1, y2 - y1) / S
            ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
            key = (round(ang / 6), round(((x1 + x2) / 2 * -math.sin(math.radians(ang)) + (y1 + y2) / 2 * math.cos(math.radians(ang))) / S / (0.05 * max(W, H))))
            found.append((L, key, (x1 / S, y1 / S, x2 / S, y2 / S, ang)))
    edm = cv2.dilate(ed, np.ones((3, 3), np.uint8)) > 0
    def supported(v):
        x1, y1, x2, y2, _ = v
        t = np.linspace(0, 1, 60)
        xs = np.clip(((x1 + (x2 - x1) * t) * S).astype(int), 0, edm.shape[1] - 1); ys = np.clip(((y1 + (y2 - y1) * t) * S).astype(int), 0, edm.shape[0] - 1)
        return edm[ys, xs].mean() >= 0.8
    seen, picks = set(), []
    for L, key, v in sorted(found, reverse=True):
        if key in seen or not supported(v) or L > 1.2 * max(W, H) * 0.99 and abs(v[4] % 90 - 45) < 40: continue
        seen.add(key); picks.append((L, v))
        if len(picks) >= 6: break
    if picks:
        out.append("LONG STRAIGHT EDGES (Hough; field boundaries, seams, panel lines, limbs of tools): " + "; ".join(
            f"{dirword(v[4])} from ({pct(v[0],W)},{pct(v[1],H)}) to ({pct(v[2],W)},{pct(v[3],H)}), length {100*L/max(W,H):.0f}%" for L, v in picks)
            + ". (Blur makes real edges look softer than ruler-straight.)")

    # ---- round features
    gn = gray8.astype(np.float64) / 255
    bl = []
    for pol, name in ((1 - gn, "dark"), (gn, "bright")):
        b = feature.blob_log(np.where(mask, pol, 0), min_sigma=max(1.0, 0.012 * W), max_sigma=0.06 * W, num_sigma=6, threshold=0.30)
        b = [x for x in b if mask[int(x[0]), int(x[1])]]
        b.sort(key=lambda x: -x[2])
        if len(b) > 12: bl.append(f"many ({len(b)}) small {name} spots (busy texture, not individual features)")
        elif b: bl.append(f"{len(b)} {name} round spot(s), largest at " + ", ".join(f"({pct(x[1],W)},{pct(x[0],H)}) r~{100*x[2]*1.41/W:.1f}%" for x in b[:5]))
    if bl:
        out.append("ROUND FEATURES (eyes, rivets, bolts, holes, highlights): " + "; ".join(bl) + ".")

    gray8_ = np.clip(lum, 0, 255).astype(np.uint8)
    words_ = ocr_boxes(gray8_, W, H)
    txt = np.zeros_like(mask)
    for w_, c_, x0_, y0_, x1_, y1_ in words_:
        txt[max(0, int(y0_) - 2):int(y1_) + 3, max(0, int(x0_) - 2):int(x1_) + 3] = True
    rows_ = stud_rows(lum, mask & ~txt)
    if rows_:
        out.append("CANDIDATE DOT ROWS (evenly spaced small dots found by measurement; they MAY be rivets, nails, studs or bolts, but regular dots are also made by windows, eyes, tile or checker patterns, markings, grain and lights, so check the picture before using them): " + "; ".join(rows_) + ".")

    # ---- symmetry / tiling
    lr = np.corrcoef(lum[:, :W // 2].ravel(), lum[:, ::-1][:, :W // 2].ravel())[0, 1]
    tb = np.corrcoef(lum[:H // 2].ravel(), lum[::-1][:H // 2].ravel())[0, 1]
    seam = (np.abs(lum[:, 0] - lum[:, -1]).mean() + 1e-9) / (np.abs(np.diff(lum, axis=1)).mean() + 1e-9)
    seam2 = (np.abs(lum[0] - lum[-1]).mean() + 1e-9) / (np.abs(np.diff(lum, axis=0)).mean() + 1e-9)
    out.append(f"SYMMETRY left-right {lr:.2f}, top-bottom {tb:.2f} (1 = mirror image). TILING seam ratio left-right {seam:.1f}, top-bottom {seam2:.1f} (about 1 or less = wraps seamlessly, much higher = a seam would show when repeated).")

    # ---- OCR for lettering (confident words only; the same words were excluded from the dot-row search)
    if words_:
        words_.sort(key=lambda t: (round(t[3] / max(1, H) * 20), t[2]))
        out.append("LETTERING read by OCR (confident words only, the redraw must keep these words exactly, spelled correctly, once, in this position): " + "; ".join(
            f'"{w}" at x {pct(x0,W)}-{pct(x1,W)}%, y {pct(y0,H)}-{pct(y1,H)}%' for w, c, x0, y0, x1, y1 in words_[:8]) + ". Words that OCR missed may still exist.")
    else:
        out.append("LETTERING: OCR found no confident words (stylised or blurry lettering can still be present).")

    # ---- overlay image for the VLM: segmentation map with a 4x4 coordinate grid
    if overlay and cents:
        S2 = 512
        pal = {i: tuple(int(v) for v in (skc.lab2rgb(c[None, None].astype(np.float64))[0, 0] * 255).round()) for i, c in enumerate(cents)}
        seg = np.full((H, W, 3), (255, 0, 255), dtype=np.uint8)
        for i, c in pal.items(): seg[lbl == i] = c
        big = cv2.resize(seg, (S2, S2), interpolation=cv2.INTER_NEAREST)
        for k in range(1, 4):
            cv2.line(big, (k * S2 // 4, 0), (k * S2 // 4, S2), (255, 255, 255), 1); cv2.line(big, (0, k * S2 // 4), (S2, k * S2 // 4), (255, 255, 255), 1)
            cv2.putText(big, f"{25*k}", (k * S2 // 4 + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(big, f"{25*k}", (2, k * S2 // 4 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        for i in pal:
            m = lbl == i
            if m.sum() == 0: continue
            dm = ndi.distance_transform_edt(m); yy, xx = np.unravel_index(dm.argmax(), dm.shape)
            cv2.putText(big, letters[i], (int((xx + .5) * S2 / W) - 6, int((yy + .5) * S2 / H) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(big, letters[i], (int((xx + .5) * S2 / W) - 6, int((yy + .5) * S2 / H) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        Image.fromarray(big).save(overlay)
    return "\n".join(out), data


if __name__ == "__main__":
    a = sys.argv[1:]
    ov = a[a.index("--overlay") + 1] if "--overlay" in a else None
    text, data = analyze(a[0], ov)
    print(text)
    if "--json" in a: json.dump(data, open(a[a.index("--json") + 1], "w"), indent=1)
