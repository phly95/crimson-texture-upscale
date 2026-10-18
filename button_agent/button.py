"""button.py: ONE renderer for the game's lettered capsule menu buttons (pass 2, plate = approved FLUX master).

    import button
    button.build("PF_B_CONTROLS")                     # -> sprites/<stem>/draw.png (what recipes/<stem>.py does)
    layer = button.render_spec(button.measure(stem))  # spec: geometry + text box measured from sprites/<stem>/orig.png

Style definition and parameters: NOTES.md. Every button is built the same way:
  1. PLATE  master_plate.png (brass state of a 221x42 button at ~5x) at frame scale kb (frame_scale(): 0.7 for 32-38 px, 0.8 for the
            MM_ 42 px family, 1.0 otherwise; the originals use these three frame weights). Horizontally the end caps are untouched and
            the two plain middle strips (above the support bar, right of the bar end) are cropped/extended with min-error-seam quilting
            so the bar ends where the original's does. Vertically a five-band map: top rim, bracket rows and bottom rim+bar keep frame
            scale; the two parchment bands take up the height. Rim rows are detected/fitted on each original (per state: EM_ sheets step).
            The original's own low-frequency parchment mottling (sigma 1 px orig, letters excluded) is carried onto the panel.
  2. STATE  states 2-4 brass as is; state 1 recolours the parchment panel only (soft mask) with one fixed Lab mean/std transfer.
  3. TEXT   Pathway Gothic One, cap height / word boxes / letter centres measured from the original's lettering pixels; fixed tracking,
            one horizontal glyph scale per button; per-state fill gradient + bevel, tight dark contour, soft dense drop shadow,
            dark halo and coloured glow (STYLE), identical on every button.
"""
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
MASTER_PATH = HERE / "master_plate.png"
FONT_PATH = str(HERE / "fonts" / "PathwayGothicOne-Regular.ttf")
FONT_WGHT = None          # weight axis value for variable fonts
K = 5                                    # target scale

# ---- master plate geometry (master pixel columns, 1104 x 208) -------------------------------------------------------------------------
M_L = 178        # end of left cap (frame curve, bracket, bar's left collar)
M_E0 = 700       # start of the bar-end zone (bar highlight still running)
M_HE = 743       # bar highlight end (measured: light run 155..742)
M_E1 = 822       # end of the bar-end zone (collar and frame step done)
M_R = 972        # start of right cap
M_H = 208

# ---- per-state style (all buttons) ----------------------------------------------------------------------------------------------------
# colours are 0..255 sRGB; fill = vertical gradient over the cap height (top, middle, bottom); lengths are in ORIGINAL pixels (x K at 5x)
STYLE = {
    0: dict(fill=[(212, 212, 210), (180, 180, 179), (160, 160, 158)], glow=None,
            shadow=(14, 11, 8), shadow_op=.95, halo_op=.80),
    1: dict(fill=[(255, 238, 176), (255, 217, 128), (252, 204, 106)], glow=((255, 214, 120), .30),
            shadow=(30, 16, 2), shadow_op=.95, halo_op=.80),
    2: dict(fill=[(255, 176, 74), (255, 150, 34), (246, 74, 0)], glow=((236, 104, 10), .55),
            shadow=(26, 4, 0), shadow_op=1.0, halo_op=.80),
    3: dict(fill=[(255, 74, 66), (248, 38, 34), (232, 8, 8)], glow=((214, 28, 16), .55),
            shadow=(22, 0, 0), shadow_op=1.0, halo_op=.80),
}
SHADOW_OFF = (0.9, 1.6)     # px (orig) right, down
SHADOW_SIGMA = 1.1         # px (orig): soft blurred shadow
SHADOW_GAIN = 2.2          # dense core, soft falloff (the originals sit the letters on a dark, soft shadow band)
EDGE_OP = .85               # tight dark contour (shadow at zero offset, sigma EDGE_SIGMA): the crisp 'cut-in' edge of the originals
EDGE_SIGMA = 0.40
HALO_SIGMA = 2.0            # px (orig): wide soft darkening round the letters (the 'sunk in' look of the originals)
GLOW_SIGMA = 0.9            # px (orig): coloured glow hugging the letters (normal blend: tints the dark halo amber / red)
WEIGHT = 0.17               # px (orig) added to each side of the glyph strokes (Pathway Gothic is a touch lighter than the original face)
BEVEL = 0.35                # bevel strength (lighter top edges, darker bottom edges)
BEVEL_SIGMA = 0.45          # px (orig)

# state-1 parchment: Lab (L, a, b) mean/std of the brass panel -> grey-beige panel (measured, see calibrate_state1)
STATE1_LAB = dict(src_mean=(60.4, 3.86, 34.2), src_std=(7.51, 0.99, 3.39),
                  dst_mean=(55.0, 0.71, 15.6), dst_std=(10.36, 0.73, 2.15))

# paper plates without lettering and without the steel frame: a different asset, not drawn from the brass master
BLANK = {"IA_B_VIEWSTORY", "IA_B_WEAPONLOADOUTFLYMISSION", "PX_B_PURCHASENOW"}

LABELS = {   # word breaks (the sprite names drop the spaces)
    "CM_B_DELETEPLAYER": "DELETE PLAYER", "EM_B_LOADGAME": "LOAD GAME", "EM_B_SAVEGAME": "SAVE GAME", "FC_B_FLYMISSION": "FLY MISSION",
    "FC_B_RETURNTOBRIEFING": "RETURN TO BRIEFING", "GN_B_BUILDCUSTOMPLANE": "BUILD CUSTOM PLANE", "GN_B_RETURNTOCABIN": "RETURN TO CABIN",
    "GN_B_RETURNTOGAME": "RETURN TO GAME", "IA_B_EXITINSTANTACTION": "EXIT INSTANT ACTION", "IA_B_FLYNEWMISSION": "FLY NEW MISSION",
    "KB_B_RESETTODEFAULT": "RESET TO DEFAULT", "LS_B_DELETEGAME": "DELETE GAME", "MM_B_INSTANTACTION": "INSTANT ACTION",
    "MP_B_EXITMULTIPLAYER": "EXIT MULTIPLAYER", "MS_B_ACCEPTCHANGES": "ACCEPT CHANGES", "MS_B_CANCELCHANGES": "CANCEL CHANGES",
    "MS_B_RETURNTOSCRAPBOOK": "RETURN TO SCRAPBOOK", "OL_B_ACCEPTLOADOUT": "ACCEPT LOADOUT", "OL_B_CANCELLOADOUT": "CANCEL LOADOUT",
    "PC_B_CHANGEMEMENTO": "CHANGE MEMENTO", "PC_B_NEXTMISSION": "NEXT MISSION", "PC_B_PLANECONSTRUCTION": "PLANE CONSTRUCTION",
    "PC_B_PREVIOUSMISSIONS": "PREVIOUS MISSIONS", "PC_B_RETURNMAINMENU": "RETURN TO MAIN MENU", "PC_B_SAVEGAME": "SAVE GAME",
    "PF_B_ACCEPTCHANGES": "ACCEPT CHANGES", "PF_B_CANCELCHANGES": "CANCEL CHANGES", "PF_B_GAMEOPTIONS": "GAME OPTIONS",
    "PF_B_MAINMENU": "MAIN MENU", "PS_B_ACCEPTSELECTIONS": "ACCEPT SELECTIONS", "PS_B_BUILDNEWPLANE": "BUILD NEW PLANE",
    "PS_B_CANCELSELECTIONS": "CANCEL SELECTIONS", "PX_B_CANCELEXPORT": "CANCEL EXPORT", "PX_B_CANCELPURCHASE": "CANCEL PURCHASE",
    "PX_B_EXPORTDESIGN": "EXPORT DESIGN", "PX_B_READYTOEXPORT": "READY TO EXPORT", "PX_B_READYTOPURCHASE": "READY TO PURCHASE",
    "PX_B_SELLPLANES": "SELL PLANES", "SB_B_EXPORTTODESKTOP": "EXPORT TO DESKTOP", "SB_B_VIEWALLMISSIONS": "VIEW ALL MISSIONS",
}


# ======================================================================================================================================
# colour helpers
def _srgb_to_lab(rgb):
    return cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)


def _lab_to_srgb(lab):
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1)


def _premul_resize(rgba, size, interp=None):
    w, h = size
    if interp is None:
        interp = cv2.INTER_AREA if w < rgba.shape[1] else cv2.INTER_LANCZOS4
    pm = np.dstack([rgba[..., :3] * rgba[..., 3:], rgba[..., 3:]])
    pm = cv2.resize(pm, (w, h), interpolation=interp)
    a = np.clip(pm[..., 3:], 0, 1)
    return np.dstack([np.clip(pm[..., :3] / np.maximum(a, 1e-5), 0, 1) * (a > 0), a])


# ======================================================================================================================================
# measuring an original
def _states(stem):
    meta = json.loads((HERE / "sprites" / stem / "meta.json").read_text())
    o = np.asarray(Image.open(HERE / "sprites" / stem / "orig.png").convert("RGBA"), np.float32)
    sh = meta["state_h"]
    return meta, o, [o[i * sh:(i + 1) * sh] for i in range(4)]


def text_mask_1x(states):
    """lettering pixels of an original: states 3 (orange) and 4 (red) share the plate, their green channel differs only in the letters"""
    a, b = states[2], states[3]
    # some sheets (EM_B_LOADGAME) step the plate by a few rows per state: align state 4 to state 3 first
    errs = {dy: np.abs(np.roll(b[..., 3], dy, 0)[6:-6] - a[6:-6, :, 3]).mean() for dy in range(-4, 5)}
    dy = min(errs, key=errs.get)
    d = a[..., 1] - np.roll(b, dy, 0)[..., 1]
    d[:max(4, dy)] = 0                       # rolled-in rows and the rims (lettering never touches the outer 4 rows)
    d[min(-4, dy):] = 0
    mk = d > 60
    lab, n = ndi.label(mk)
    if n == 0:
        return mk
    sizes = ndi.sum(mk, lab, range(1, n + 1))
    # drop specks and flat wide streaks (plate edges left by an imperfect state alignment): letters are never <= 2 rows tall
    sl = ndi.find_objects(lab)
    keep_ids = [i + 1 for i, (sz, o) in enumerate(zip(sizes, sl))
                if sz >= 4 and not ((o[0].stop - o[0].start) <= 2 and (o[1].stop - o[1].start) > 15)]
    mk = np.isin(lab, keep_ids)
    rows = mk.sum(1)
    if rows.max() == 0:
        return mk
    # keep the main band of rows (the text line), drop stray differences on the frame
    good = rows >= max(2, .12 * rows.max())
    lab1, _ = ndi.label(good)
    keep = lab1 == lab1[int(np.argmax(rows))]
    mk[~keep] = False
    return mk


def bar_highlight_end(rgba):
    """x (exclusive) where the support bar's light highlight line ends"""
    a = rgba.astype(np.float32); h, w = a.shape[:2]
    lum = a[..., :3].mean(-1); sat = a[..., :3].max(-1) - a[..., :3].min(-1)
    g = (lum > 110) & (sat < 30) & (a[..., 3] > 200)
    best = (0, None)
    for y in range(int(.7 * h), h):
        lab, n = ndi.label(g[y])
        for i in range(1, n + 1):
            xs = np.where(lab == i)[0]
            if xs[-1] - xs[0] > best[0]:
                best = (xs[-1] - xs[0], xs[-1] + 1)
    return best[1]


def measure(stem):
    """spec of a button from its original: plate box, bar end, text box, word boxes (all in original pixels)"""
    meta, o, st = _states(stem)
    sh = meta["state_h"]; W = o.shape[1]
    al = st[1][..., 3] > 127
    ys, xs = np.where(al)
    tm = text_mask_1x(st)
    words = []
    label = LABELS.get(stem, meta["label"])
    if tm.any():
        ty, tx = np.where(tm)
        cols = tm.any(0)
        # word boxes: split the column runs at the (nwords-1) widest gaps
        runs = []; x = 0
        while x < W:
            if cols[x]:
                x2 = x
                while x2 < W and cols[x2]:
                    x2 += 1
                runs.append([x, x2]); x = x2
            else:
                x += 1
        nw = len(label.split())
        gaps = sorted(range(len(runs) - 1), key=lambda i: runs[i + 1][0] - runs[i][1], reverse=True)[:nw - 1]
        cut = sorted(gaps)
        start = 0
        letters = []
        for c in cut + [len(runs) - 1]:
            words.append((runs[start][0], runs[c][1]))
            letters.append([((a + b) / 2) for a, b in runs[start:c + 1]])
            start = c + 1
        text = dict(x0=int(tx.min()), x1=int(tx.max() + 1), y0=int(ty.min()), y1=int(ty.max() + 1), words=words, letters=letters)
    else:
        text = None
    be = bar_highlight_end(st[1])
    kb = frame_scale(sh, stem)
    rims = [fit_rims(st[i], be, text, kb) for i in range(4)]
    D = float(np.median([r[1] - r[0] for r in rims]))         # same plate in all four states: one rim distance / frame scale per button
    kb = float(np.median([r[2] for r in rims]))
    rims = [(r[0], round(r[0] + D, 2), kb) for r in rims]
    return dict(stem=stem, label=label, W=W, H=o.shape[0], sh=sh,
                plate=dict(x0=int(xs.min()), x1=int(xs.max() + 1), y0=int(ys.min()), y1=int(ys.max() + 1)),
                bar_end=be, rims=rims, text=text)


# ---- vertical plate map: master row r -> state row y (original px). The steel tube and the bar keep their size on every button
# (the originals use the same pixel pattern at every height); only the parchment band between master rows M_RT and M_RB is resized.
M_TC, M_BC = 9.75, 182.5      # master rows of the top / bottom rim highlight (right of the bar)
TC_OFFSET = 0.0               # detected rim row -> map coordinate (empirical: best silhouette and colour match)
M_RT, M_RB = 40, 140          # fixed top band [0, M_RT), fixed bottom band [M_RB, 208)
KM = 210 / 208                # target px per master px (master = 42 px design at 5x)
M_BK0, M_BK1, M_BKC = 74, 124, 99   # master rows of the side brackets (kept at frame scale, never stretched)


def frame_scale(sh, stem=""):
    """the originals use three frame weights; the plate keeps the master at this scale (tube, brackets, bar, cap width):
    thin 0.7 for 32-38 px states, 0.8 for the 42 px main-menu family (MM_: thin tube, plate 40 px), full 1.0 for the other 42-50 px"""
    if sh < 40:
        return 0.7
    if stem.startswith("MM_") and sh < 46:
        return 0.8
    return 1.0


def _r_of_y(y5, tc, bc, kb=1.0):
    """inverse of the vertical map: 5x state row(s) -> master row(s) (tc, bc: rim highlight rows in original px; kb: frame scale).
    Five bands: top rim [0, M_RT), bottom rim+bar [M_RB, end) and the brackets [M_BK0, M_BK1) at frame scale; the two parchment
    bands between them take up the height difference."""
    km = KM * kb
    y5 = np.asarray(y5, np.float32)
    tc = np.asarray(tc, np.float32); bc = np.asarray(bc, np.float32)
    yT = 5 * tc + (M_RT - M_TC) * km
    yB = 5 * bc + (M_RB - M_BC) * km
    c = yT + (M_BKC - M_RT) / (M_RB - M_RT) * (yB - yT)
    y0b = c - (M_BKC - M_BK0) * km; y1b = c + (M_BK1 - M_BKC) * km
    top = M_TC + (y5 - 5 * tc) / km
    bot = M_BC + (y5 - 5 * bc) / km
    up = M_RT + (y5 - yT) * (M_BK0 - M_RT) / np.maximum(1e-3, y0b - yT)
    br = M_BKC + (y5 - c) / km
    lo = M_BK1 + (y5 - y1b) * (M_RB - M_BK1) / np.maximum(1e-3, yB - y1b)
    return np.where(y5 < yT, top, np.where(y5 < y0b, up, np.where(y5 < y1b, br, np.where(y5 < yB, lo, bot))))


_PROF = None


def _master_profiles():
    global _PROF
    if _PROF is None:
        M = _master() * 255
        pm = M[..., :3].mean(-1) * M[..., 3] / 255
        _PROF = {k: (pm[:, a:b].mean(1), M[:, a:b, 3].mean(1)) for k, (a, b) in {"bar": (320, 600), "nobar": (840, 950)}.items()}
    return _PROF


def fit_rims(stt, be, text, kb=1.0, with_cost=False):
    """rows (original px, state coords) of the top and bottom rim highlights that make the mapped master's vertical profile match
    this state of the original best (bar columns and columns right of the bar end; text rows ignored)"""
    h, w = stt.shape[:2]
    pm = stt[..., :3].mean(-1) * stt[..., 3] / 255
    regions = {"bar": (int(.35 * be), int(.75 * be)), "nobar": (be + 6, w - 26)}
    wt = np.ones(h, np.float32)
    if text:
        wt[max(0, text["y0"] - 2):text["y1"] + 2] = 0
    # ignore anything below this plate's own bottom (some sheets show the next plate's rim there)
    na, nb_ = regions["nobar"] if regions["nobar"][1] - regions["nobar"][0] >= 3 else regions["bar"]
    am = stt[:, na:nb_, 3].mean(1)
    rt = next((r for r in range(h // 2, h) if am[r] < 80), h)
    wt[rt + 1:] = 0
    prof = _master_profiles()
    # the top rim highlight is a narrow bright row (e.g. 43 96 43): detect it directly where it is visible
    a, b = regions["nobar"] if regions["nobar"][1] - regions["nobar"][0] >= 3 else regions["bar"]
    Lm = np.median(pm[:, a:b], 1)
    tc_fix = None
    for r in range(0, 7):
        lo = Lm[r - 1] if r > 0 else -1e9
        if Lm[r] > lo + 20 and Lm[r] > Lm[r + 1] + 20 and (r > 0 or stt[0, a:b, 3].mean() > 128):
            d = (Lm[r - 1] - 2 * Lm[r] + Lm[r + 1]) if r > 0 else 0
            tc_fix = r + TC_OFFSET + (.5 * (Lm[r - 1] - Lm[r + 1]) / d if d else 0.)
            break
    sub = 8
    t = (np.arange(h * sub, dtype=np.float32) + .5) / sub
    best = (np.inf, None)
    for tc, kb in [(a_, b_) for a_ in ([tc_fix] if tc_fix is not None else np.arange(-4, 5.01, .1)) for b_ in (kb,)]:
        bcs = np.arange(tc + 18, tc + 50, .1)
        r = _r_of_y(5 * t[None, :], tc, bcs[:, None], kb)                    # (nbc, h*sub)
        cost = np.zeros(len(bcs), np.float32)
        for k, (a, b) in regions.items():
            if b - a < 3:
                continue
            ol = pm[:, a:b].mean(1); oa = stt[:, a:b, 3].mean(1)
            P, A = prof[k]
            pl = np.interp(r, np.arange(len(P)), P, left=0, right=0).reshape(len(bcs), h, sub).mean(2)
            pa = np.interp(r, np.arange(len(A)), A, left=0, right=0).reshape(len(bcs), h, sub).mean(2)
            cost += (((pl - ol) ** 2 + (pa - oa) ** 2) * wt).sum(1)
        i = int(np.argmin(cost))
        if cost[i] < best[0]:
            best = (cost[i], (round(float(tc), 2), round(float(bcs[i]), 2), kb))
    return (float(best[0]), best[1]) if with_cost else best[1]


# ======================================================================================================================================
# plate
_MASTER = None


def _master():
    global _MASTER
    if _MASTER is None:
        _MASTER = np.asarray(Image.open(MASTER_PATH).convert("RGBA"), np.float32) / 255
    return _MASTER


def _seam_join(P, Q, ov):
    """join P (left) and Q (right) overlapping by ov columns along a minimum-error vertical seam, softly feathered"""
    if ov <= 0:
        return np.concatenate([P, Q], 1)
    a = P[:, -ov:]; b = Q[:, :ov]
    pa = np.dstack([a[..., :3] * a[..., 3:], a[..., 3:]]); pb = np.dstack([b[..., :3] * b[..., 3:], b[..., 3:]])
    e = ((pa - pb) ** 2).sum(-1)
    h = e.shape[0]
    cost = e.copy(); back = np.zeros_like(e, np.int32)
    for y in range(1, h):
        prev = cost[y - 1]
        l = np.r_[np.inf, prev[:-1]]; r = np.r_[prev[1:], np.inf]
        stack = np.vstack([l, prev, r]); i = stack.argmin(0)
        cost[y] += stack[i, np.arange(ov)]; back[y] = i - 1
    path = np.zeros(h, np.int32); path[-1] = int(cost[-1].argmin())
    for y in range(h - 1, 0, -1):
        path[y - 1] = np.clip(path[y] + back[y, path[y]], 0, ov - 1)
    xx = np.arange(ov)[None, :]
    m = (xx > path[:, None]).astype(np.float32)               # 0 = take P, 1 = take Q
    m = cv2.GaussianBlur(m, (0, 0), 1.2)
    pm = pa * (1 - m[..., None]) + pb * m[..., None]
    al = np.clip(pm[..., 3:], 0, 1)
    mid = np.dstack([pm[..., :3] / np.maximum(al, 1e-5), al])
    return np.concatenate([P[:, :-ov], mid, Q[:, ov:]], 1)


def _synth(S, n, ov):
    """n columns that start like S's first column and end like S's last one (crop or extend S invisibly)"""
    ns = S.shape[1]
    if n == ns:
        return S
    if n <= 0:
        return S[:, :0]
    if n < ns:
        ov = int(min(ov, n // 2, ns - (n + 1) // 2))
        if ov < 2:
            # too short for a seam: plain crop from the start + resample (only for tiny strips)
            return _premul_resize(S, (n, S.shape[0]))
        a = (n + ov) // 2; b = n + ov - a
        return _seam_join(S[:, :a], S[:, ns - b:], ov)
    if n <= 2 * ns - ov:
        c = 2 * ns - ov - n
        return _seam_join(S, S[:, c:], ov)
    return _seam_join(S, _synth(S, n - ns + ov, ov), ov)


def fit_plate(tc, bc, kb, Wp, bar_end):
    """plate for one state: (rgba, y0) with rgba rows = 5x state rows y0.. ; Wp wide, bar highlight ending at bar_end (5x, plate-relative).
    The master is used at scale KM*kb horizontally (caps, brackets, bar untouched); vertically the frame bands keep that scale and only
    the parchment band is resized."""
    M = _master()
    km = KM * kb
    Wm = int(round(M.shape[1] * km))
    Ms = _premul_resize(M, (Wm, M.shape[0]))
    c = lambda x: int(round(x * km))
    L, E0, E1, R, HE = c(M_L), c(M_E0), c(M_E1), c(M_R), c(M_HE)
    ov = int(40 * kb)
    tE0 = int(round(bar_end - (HE - E0)))
    nA = tE0 - L
    nE = E1 - E0
    nR = Wm - R
    nB = Wp - nR - (tE0 + nE)
    if nB < 2 or nA < 2:
        raise ValueError(f"plate too short for the bar layout (nA={nA}, nB={nB})")
    A = _synth(Ms[:, L:E0], nA, ov)
    B = _synth(Ms[:, E1:R], nB, ov)
    wide = np.concatenate([Ms[:, :L], A, Ms[:, E0:E1], B, Ms[:, R:]], 1)
    # vertical: remap rows through the piecewise map (premultiplied); low-pass first where the parchment band is squashed
    y_top = 5 * tc + (0 - M_TC) * km; y_bot = 5 * bc + (M.shape[0] - M_BC) * km
    y0 = int(np.floor(y_top)); y1 = int(np.ceil(y_bot))
    rows = _r_of_y(np.arange(y0, y1, dtype=np.float32) + .5, tc, bc, kb) - .5
    pm = np.dstack([wide[..., :3] * wide[..., 3:], wide[..., 3:]]).astype(np.float32)
    mid_scale = ((5 * bc + (M_RB - M_BC) * km) - (5 * tc + (M_RT - M_TC) * km) - (M_BK1 - M_BK0) * km) / (M_RB - M_RT - (M_BK1 - M_BK0))
    if mid_scale < .95:
        sig = .5 * np.sqrt(max(0., 1 / mid_scale ** 2 - 1))
        blurred = ndi.gaussian_filter1d(pm, sig, axis=0)
        wgt = np.clip((np.arange(pm.shape[0]) - (M_RT - 4)) / 4, 0, 1) * np.clip(((M_RB + 4) - np.arange(pm.shape[0])) / 4, 0, 1)
        pm = pm * (1 - wgt[:, None, None]) + blurred * wgt[:, None, None]
    mx, my = np.meshgrid(np.arange(Wp, dtype=np.float32), rows.astype(np.float32))
    out = cv2.remap(pm, mx, my, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    al = np.clip(out[..., 3:], 0, 1)
    out = np.dstack([np.clip(out[..., :3] / np.maximum(al, 1e-5), 0, 1) * (al > 0), al])
    assert out.shape[1] == Wp, (out.shape, Wp)
    return out, y0


_PANEL = {}


def panel_mask(plate):
    """soft mask of the parchment panel (everything inside the steel frame that is not steel)"""
    rgb = plate[..., :3]; a = plate[..., 3]
    lab = _srgb_to_lab(rgb)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    h, w = a.shape
    core = (chroma > 9) & (a > .99)
    core = ndi.binary_opening(core, iterations=2)
    lab_, n = ndi.label(core)
    if n:
        sizes = ndi.sum(core, lab_, range(1, n + 1))
        core = lab_ == (1 + int(np.argmax(sizes)))
    core = ndi.binary_fill_holes(ndi.binary_closing(core, iterations=6))
    # include the dark inner rim of the parchment (warm shadow next to the frame) but no neutral steel
    grow = ndi.binary_dilation(core, iterations=max(2, int(h * .02))) & (chroma > 5) & (a > .99)
    m = (core | grow).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), max(1., h * .006))


MOTTLE_SIGMA = 1.0    # px (orig): low-pass of the original's own parchment mottling carried onto the plate
MOTTLE_STRENGTH = 0.8


def transfer_mottling(plate, py0, x0, ref, text_mask, tc_shift):
    """carry the ORIGINAL's own low-frequency parchment mottling (light/dark blotch layout, sigma MOTTLE_SIGMA orig px) onto the panel;
    the master's HD grain, filigree and frame stay. The original's lettering zone is excluded (no ghost letters).
    plate: fitted plate (5x), placed at state rows py0.., cols x0..; ref: the original's brass state (1x RGBA, float 0..255)."""
    h, w = plate.shape[:2]
    H1, W1 = ref.shape[:2]
    # original brass state at 5x in plate coordinates
    up = cv2.resize(ref[..., :3] / 255, (W1 * K, H1 * K), interpolation=cv2.INTER_CUBIC)
    ua = cv2.resize(ref[..., 3] / 255, (W1 * K, H1 * K), interpolation=cv2.INTER_LINEAR)
    oy = py0 + int(round(tc_shift * K))      # reference rows sit tc_shift (orig px) lower than this state's
    ref5 = np.zeros((h, w, 3), np.float32); refa = np.zeros((h, w), np.float32)
    ya, yb = max(0, oy), min(H1 * K, oy + h)
    if yb <= ya:
        return plate
    ref5[ya - oy:yb - oy] = up[ya:yb, x0:x0 + w]; refa[ya - oy:yb - oy] = ua[ya:yb, x0:x0 + w]
    pm = panel_mask(plate)
    core = (cv2.erode((pm > .9).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=int(2 * K)) > 0) & (refa > .99)
    m = core.astype(np.float32)
    if text_mask is not None:
        # exclude the original's letters with their glow/shadow (dilated, extended downwards), not the whole text box
        tm = ndi.binary_dilation(text_mask, iterations=2)
        tm = tm | np.roll(tm, 1, 0) | np.roll(tm, 2, 0)
        tz = cv2.resize(tm.astype(np.float32), (W1 * K, H1 * K), interpolation=cv2.INTER_NEAREST)
        tzp = np.zeros((h, w), np.float32)
        tzp[ya - oy:yb - oy] = tz[ya:yb, x0:x0 + w]
        m = m * (1 - tzp)
    sig = MOTTLE_SIGMA * K
    lab_o = _srgb_to_lab(ref5); lab_p = _srgb_to_lab(plate[..., :3])
    nb = lambda x, mm: cv2.GaussianBlur(x * mm[..., None], (0, 0), sig) / np.maximum(cv2.GaussianBlur(mm, (0, 0), sig), 1e-4)[..., None]
    d = nb(lab_o, m) - nb(lab_p, m)
    # weight: where the original was observed (feathered), zero elsewhere (text zone, frame)
    wgt = cv2.GaussianBlur(m, (0, 0), 2 * sig) * pm
    wgt = np.clip(wgt * 1.3, 0, 1) * MOTTLE_STRENGTH
    lab = lab_p + d * wgt[..., None]
    out = plate.copy(); out[..., :3] = _lab_to_srgb(lab)
    return out


def recolour_state1(plate, mask):
    lab = _srgb_to_lab(plate[..., :3])
    sm, ss = np.array(STATE1_LAB["src_mean"]), np.array(STATE1_LAB["src_std"])
    dm, ds = np.array(STATE1_LAB["dst_mean"]), np.array(STATE1_LAB["dst_std"])
    new = (lab - sm) * (ds / ss) + dm
    rgb = _lab_to_srgb(new)
    out = plate.copy()
    out[..., :3] = plate[..., :3] * (1 - mask[..., None]) + rgb * mask[..., None]
    return out


# ======================================================================================================================================
# lettering
SS = 3          # supersampling of the glyph coverage


def _font(px_cap):
    """font whose cap height is px_cap pixels (at the supersampled scale)"""
    def mk(sz):
        f = ImageFont.truetype(FONT_PATH, sz)
        if FONT_WGHT is not None:
            f.set_variation_by_axes([FONT_WGHT])
        return f
    b = mk(400).getbbox("H")
    return mk(max(4, int(round(400 * px_cap / (b[3] - b[1])))))


ALIGN = 0.5         # 0 = evenly tracked layout, 1 = every glyph centred on the original letter (where letters are separable)
CAP_TRIM = 0.0      # measured text rows include ~1 anti-aliased row: cap height = (y1 - y0 - CAP_TRIM) orig px


TRACK = 0.09        # standard letter spacing (fraction of the cap height) on every button
TRACK_RANGE = (0.04, 0.16)
SX_RANGE = (0.88, 1.15)   # horizontal glyph scale that absorbs the rest of the difference to the original's text width


def text_layout(spec):
    """glyph coverage (target size of one state) for spec['label'] fitted to spec['text'] box and word positions.
    Every button uses tracking TRACK x cap; the remaining difference to the original's word widths is taken up by one horizontal
    glyph scale per button (SX_RANGE), then by tracking within TRACK_RANGE; words sit at the original's word centres."""
    t = spec["text"]; W5, H5 = spec["W"] * K, spec["sh"] * K
    cap = (t["y1"] - t["y0"] - CAP_TRIM) * K * SS
    f = _font(cap)
    hb = f.getbbox("H")
    words = spec["label"].split()
    boxes = t["words"] if len(t["words"]) == len(words) else [(t["x0"], t["x1"])]
    if len(boxes) != len(words):
        words = [" ".join(words)]

    def ink(word, tr):
        xs = 0.; first = f.getbbox(word[0])[0]; last = 0.
        for i, ch in enumerate(word):
            if i == len(word) - 1:
                last = xs + f.getbbox(ch)[2]
            xs += f.getlength(ch) + tr
        return last - first
    target = sum((b - a) * K * SS for a, b in boxes)
    n_gaps = sum(len(w) - 1 for w in words)
    nat0 = sum(ink(w, TRACK * cap) for w in words)
    sx = float(np.clip(target / nat0, *SX_RANGE))
    tr = (target / sx - sum(ink(w, 0) for w in words)) / max(n_gaps, 1)
    tr = float(np.clip(tr, TRACK_RANGE[0] * cap, TRACK_RANGE[1] * cap))
    # draw in unscaled space (x / sx), then stretch horizontally by sx
    Wu = int(np.ceil(W5 * SS / sx)) + 2
    im = Image.new("L", (Wu, H5 * SS), 0)
    d = ImageDraw.Draw(im)
    ytop = (t["y0"] + CAP_TRIM / 2) * K * SS - hb[1]
    letters = t.get("letters") if len(boxes) == len(t["words"]) else None
    sw = int(round(WEIGHT * K * SS / sx))
    for wi_, (w, (a, b)) in enumerate(zip(words, boxes)):
        wi = ink(w, tr)
        cx = (a + b) / 2 * K * SS / sx
        x = cx - wi / 2 - f.getbbox(w[0])[0]
        cent = letters[wi_] if letters and len(letters[wi_]) == len(w) else None
        for i, ch in enumerate(w):
            xx = x
            if cent is not None:     # pull each glyph part way to the original letter's centre (the original's own rhythm)
                bb = f.getbbox(ch)
                xo = (cent[i] + .5) * K * SS / sx - (bb[0] + bb[2]) / 2
                xx = x + ALIGN * (xo - x)
            d.text((xx, ytop), ch, font=f, fill=255, stroke_width=sw, stroke_fill=255)
            x += f.getlength(ch) + tr
    a_ = np.asarray(im, np.float32)
    a_ = cv2.resize(a_, (int(round(Wu * sx)), H5 * SS), interpolation=cv2.INTER_AREA if sx < 1 else cv2.INTER_CUBIC)
    a_ = np.clip(a_[:, :W5 * SS], 0, 255)
    if a_.shape[1] < W5 * SS:
        a_ = np.pad(a_, ((0, 0), (0, W5 * SS - a_.shape[1])))
    cov = a_.reshape(H5, SS, W5, SS).mean((1, 3)) / 255
    return cov, dict(tracking=tr * sx / SS, cap=cap / SS, sx=sx)


def _shift(x, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(x, M, (x.shape[1], x.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def letter_state(base, cov, s, ytop, ybot):
    """composite the lettering of state s onto base (H, W, 4 straight rgba)"""
    st = STYLE[s]
    H, W = cov.shape
    out = base[..., :3].copy()
    g = lambda s_: max(.3, s_ * K)
    # 1. halo: wide soft darkening (letters look pressed into the plate)
    halo = cv2.GaussianBlur(_shift(cov, .4 * K, .6 * K), (0, 0), g(HALO_SIGMA))
    halo = np.clip(halo * 1.8, 0, 1) * st["halo_op"]
    sc = np.array(st["shadow"], np.float32) / 255
    out = out * (1 - halo[..., None]) + sc * halo[..., None]
    # 2. coloured glow: normal blend of the glow colour, strongest at the letter edge
    if st["glow"]:
        gc, gop = st["glow"]
        gl = cv2.GaussianBlur(cov, (0, 0), g(GLOW_SIGMA))
        gl = np.clip(gl * 1.8, 0, 1) * gop
        gc = np.array(gc, np.float32) / 255
        out = out * (1 - gl[..., None]) + gc * gl[..., None]
    # 3. soft drop shadow, small offset
    sh = cv2.GaussianBlur(_shift(cov, SHADOW_OFF[0] * K, SHADOW_OFF[1] * K), (0, 0), g(SHADOW_SIGMA))
    sh = np.clip(sh * SHADOW_GAIN, 0, 1) * st["shadow_op"]
    out = out * (1 - sh[..., None]) + sc * sh[..., None]
    # 3b. tight dark contour
    ed = cv2.GaussianBlur(_shift(cov, .15 * K, .25 * K), (0, 0), g(EDGE_SIGMA))
    ed = np.clip(ed * 1.6, 0, 1) * EDGE_OP
    out = out * (1 - ed[..., None]) + sc * ed[..., None]
    # 4. letter fill: vertical gradient top -> middle -> bottom of the cap height
    yy = np.clip((np.arange(H, dtype=np.float32) - ytop) / max(1., ybot - ytop), 0, 1)[:, None]
    c0, c1, c2 = [np.array(c, np.float32) / 255 for c in st["fill"]]
    t1 = np.clip(yy * 2, 0, 1)[..., None]; t2 = np.clip(yy * 2 - 1, 0, 1)[..., None]
    fill = np.where(yy[..., None] < .5, c0 * (1 - t1) + c1 * t1, c1 * (1 - t2) + c2 * t2)
    fill = np.broadcast_to(fill, (H, W, 3))
    # bevel: light from the top-left on the stroke edges
    sm = cv2.GaussianBlur(cov, (0, 0), g(BEVEL_SIGMA))
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3) / 8; gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3) / 8
    light = (gy * .85 + gx * .45) * (2.2 * K / 5)                 # >0 on top/left edges
    light = np.clip(light * 3.0, -1, 1) * BEVEL
    fill = np.where(light[..., None] > 0, fill + (1 - fill) * light[..., None], fill * (1 + .6 * light[..., None]))
    out = out * (1 - cov[..., None]) + fill * cov[..., None]
    return np.dstack([np.clip(out, 0, 1), base[..., 3:]])


# ======================================================================================================================================
def render_spec(spec):
    W5, H5, sh5 = spec["W"] * K, spec["H"] * K, spec["sh"] * K
    p = spec["plate"]
    Wp = (p["x1"] - p["x0"]) * K
    x0 = p["x0"] * K
    if spec.get("text"):
        cov, info = text_layout(spec)
        t = spec["text"]
        ytop = (t["y0"] + CAP_TRIM / 2) * K; ybot = (t["y1"] - CAP_TRIM / 2) * K
    out = np.zeros((H5, W5, 4), np.float32)
    PAD = 40
    ref = None
    if spec.get("stem") and (HERE / "sprites" / spec["stem"] / "orig.png").exists() and MOTTLE_STRENGTH > 0:
        _st = _states(spec["stem"])[2]
        ref = _st[1]
        ref_tm = text_mask_1x(_st)
        # text_mask_1x is in state-3 coordinates; the brass reference is state 2: shift by their plate offset
        ref_tm = np.roll(ref_tm, int(round(spec["rims"][1][0] - spec["rims"][2][0])), 0)
    for s in range(4):
        tc, bc, kb = spec["rims"][s]
        plate, py0 = fit_plate(tc, bc, kb, Wp, spec["bar_end"] * K - x0)
        if ref is not None:
            plate = transfer_mottling(plate, py0, x0, ref, ref_tm, spec["rims"][1][0] - tc)
        if s == 0:
            plate = recolour_state1(plate, panel_mask(plate))
        # state canvas with PAD rows above/below (plates may straddle the state boundary, like EM_B_LOADGAME's)
        layer = np.zeros((sh5 + 2 * PAD, W5, 4), np.float32)
        ya = py0 + PAD
        a0, a1 = max(0, ya), min(layer.shape[0], ya + plate.shape[0])
        layer[a0:a1, x0:x0 + Wp] = plate[a0 - ya:a1 - ya]
        if spec.get("text"):
            dy = int(round((tc - spec["rims"][2][0]) * K))          # lettering moves with its plate (text box measured in state 3)
            cpad = np.zeros(layer.shape[:2], np.float32); cpad[PAD + dy:PAD + dy + sh5] = cov
            layer = letter_state(layer, cpad, s, ytop + PAD + dy, ybot + PAD + dy)
        # composite over the sheet (premultiplied 'over'), clipped to the sheet
        g0 = s * sh5 - PAD
        b0, b1 = max(0, g0), min(H5, g0 + layer.shape[0])
        src = layer[b0 - g0:b1 - g0]; dst = out[b0:b1]
        sa = src[..., 3:]; da = dst[..., 3:]
        oa = sa + da * (1 - sa)
        orgb = (src[..., :3] * sa + dst[..., :3] * da * (1 - sa)) / np.maximum(oa, 1e-5)
        out[b0:b1] = np.dstack([orgb * (oa > 0), oa])
    for w in (.2, .3, .4, .5):              # smallest edge blend that satisfies the checker's tile-seam metric
        res = _edge_wrap(out, w)
        if _seam(res) < 1.35:
            break
    return res


def _seam(layer):
    """the checker's wrap-around discontinuity metric (sprite_tool.seam) on the 8-bit result"""
    a = (np.clip(layer, 0, 1) * 255 + .5).astype(np.uint8).astype(np.float32) / 255
    a = np.dstack([a[..., :3] * a[..., 3:], a[..., 3:]])
    r = []
    for ax in (0, 1):
        edge = np.abs(np.take(a, 0, ax) - np.take(a, -1, ax)).mean()
        inner = np.abs(np.diff(a, axis=ax)).mean()
        r.append(edge / max(inner, 1e-4))
    return float(np.mean(r))


def _edge_wrap(out, w=.2):
    """The checker treats these sheets as tiling (the original's edges happen to wrap) and compares the outermost rows/columns relative
    to the neighbour differences, which are far smaller at 5x. Make the two outer columns identical (both are the bracket's outer edge)
    and pull the first and last rows w of the way towards each other (premultiplied). 1 px at 5x: invisible, 0.2 px in game."""
    pm = np.dstack([out[..., :3] * out[..., 3:], out[..., 3:]])
    c = (pm[:, 0] + pm[:, -1]) / 2
    pm[:, 0] = c; pm[:, -1] = c
    r0, r1 = pm[0].copy(), pm[-1].copy()
    pm[0] = r0 * (1 - w) + r1 * w; pm[-1] = r1 * (1 - w) + r0 * w
    a = np.clip(pm[..., 3:], 0, 1)
    return np.dstack([np.clip(pm[..., :3] / np.maximum(a, 1e-5), 0, 1), a])


def render(stem):
    return render_spec(measure(stem))


def save(layer, path):
    q = (np.clip(layer, 0, 1) * 255 + .5).astype(np.uint8)
    q[..., :3] *= (q[..., 3:] > 0)             # colour under (quantised) alpha 0 is black, never a stray key colour
    Image.fromarray(q, "RGBA").save(path)


def build(stem):
    """render sprites/<stem>/draw.png (used by recipes/<stem>.py)"""
    layer = render(stem)
    save(layer, HERE / "sprites" / stem / "draw.png")
    return layer
