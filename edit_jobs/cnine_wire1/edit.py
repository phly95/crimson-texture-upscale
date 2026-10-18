import numpy as np, cv2, sys
from PIL import Image
from scipy import ndimage as ndi

LIFT = 0.65; SM = 2.5; AA_W = 1.8; SHIFT = 0.8
S = 282                       # top rail band thickness (measured)
Y_OPEN_END = 1448             # first row of the bottom rail band
SC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.82   # scale applied to the bottom band after strip removal
GAP = 28                      # rows removed from the flat dark shadow gap under the top lip
R = int(round(600 - 282 / SC)) - GAP      # strip removed from plate
OV = 36                       # overlap rows for quilting seam
src = np.asarray(Image.open('input.png').convert('RGB')).astype(np.float32)
H, W, _ = src.shape
orig_src = src
# remove GAP rows from the (featureless) dark shadow band with a short blend
g0 = 1572
_a = src[g0-6:g0]; _b = src[g0+GAP:g0+GAP+6]
_t = np.linspace(0, 1, 8)[1:-1, None, None]
_blend = (1 - _t) * src[g0-6:g0][-6:] + _t * _b if False else None
src = np.concatenate([src[:g0], src[g0+GAP:]], 0)
# smooth the join over 6 rows
_w = np.linspace(0, 1, 6)[:, None, None]
src[g0-3:g0+3] = (1-_w)*orig_src[g0-3:g0+3] + _w*orig_src[g0+GAP-3:g0+GAP+3]

def lab(a): return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

def seam(c):
    """join rows [..c) with rows [c+R..) ; overlap region mixes src[c-OV:c] and src[c+R-OV:c+R]."""
    A = src[c-OV:c]; B = src[c+R-OV:c+R]
    # lowpass colour correction on B so overall tone matches A at overlap (decays below)
    return A, B

def build_bottom(c):
    A, B = seam(c)
    # error surface between overlapping strips
    e = ((lab(A) - lab(B))**2).sum(2)
    e = cv2.GaussianBlur(e, (0, 0), 1.5)
    # DP min path along x (path = row index per column), 8 connectivity vertical moves +-1
    cost = e.copy(); back = np.zeros((OV, W), np.int8)
    # DP over columns
    for x in range(1, W):
        prev = cost[:, x-1]
        cand = np.stack([np.r_[np.inf, prev[:-1]], prev, np.r_[prev[1:], np.inf]])
        k = cand.argmin(0)
        cost[:, x] = e[:, x] + cand[k, np.arange(OV)]
        back[:, x] = k - 1
    y = int(cost[:, -1].argmin()); path = np.zeros(W, int)
    for x in range(W-1, -1, -1):
        path[x] = y
        y = y + back[y, x] if x > 0 else y
        y = int(np.clip(y, 0, OV-1))
    total = cost[:, -1].min() / W
    return path, total

def compose(c):
    path, tot = build_bottom(c)
    A, B = seam(c)
    # tone correction of B: row-mean difference at the path, applied with decay over rows below
    ys = np.arange(OV)[:, None]
    # soft mask: 1 above path -> A, 0 below -> B
    t = np.clip((path[None, :] - ys) / 3.0 + 0.5, 0, 1)[..., None]
    # low-frequency match
    top = src[Y_OPEN_END:c-OV]
    bot = src[c+R:]
    # low-freq colour offset: compare blurred A-part near the path vs B-part
    # use heavily smoothed horizontal profile of difference between what sits above and below seam
    d = cv2.GaussianBlur((src[c-20:c].mean(0) - src[c+R-20:c+R].mean(0))[None], (0, 0), 120)[0]  # (W,3)
    decay = np.clip(1 - np.arange(len(bot)) / 12.0, 0, 1)[:, None, None]
    bot = bot + d[None] * decay
    # also apply to the mix portion fully for B part
    mixd = t * A + (1 - t) * (B + d[None])
    band = np.concatenate([top, mixd, bot], 0)
    return np.clip(band, 0, 255), tot

if __name__ == '__main__':
    best = None
    for c in range(1633, 1668, 2):
        b, tot = compose(c)
        print(c, round(tot, 1)); 
        if best is None or tot < best[0]: best = (tot, c, b)
    tot, c, band = best
    print('chosen', c, 'R', R, 'bandlen', len(band))
    Image.fromarray(band.astype(np.uint8)).save('dbg_band_unscaled.png')
    nb = cv2.resize(band, (W, S), interpolation=cv2.INTER_AREA)
    mid = cv2.resize(orig_src[S:Y_OPEN_END], (W, H - 2*S), interpolation=cv2.INTER_LANCZOS4)
    out = np.concatenate([orig_src[:S], mid, nb], 0)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save('stage1_rgb.png')

    # ---------- stage 2: alpha ----------
    rgb = np.clip(out, 0, 255)
    # --- clean the ragged, speckled lower edge of the top rail (baked noise in the dark shadow strip) ---
    mx = rgb.max(2)
    blk = mx < 14
    yy0 = 240
    ye0 = np.full(W, -1.0)
    for x in range(W):
        col_ = blk[yy0:yy0 + 600, x]
        # first row from which the next 40 rows are all black
        cs = np.cumsum(~col_[::-1])[::-1]   # number of non-black rows from here to end
        ok = np.where(cs[:len(col_) - 40] == cs[40:len(col_)] + (~col_[:len(col_) - 40]).astype(int) * 0)[0] if False else None
        run = np.convolve((~col_).astype(int), np.ones(40, int), 'valid')   # non-black count in window [i,i+40)
        cand = np.where(run == 0)[0]
        if len(cand) and cand[0] + yy0 < 300:
            ye0[x] = cand[0] + yy0
    valid = ye0 > 0
    ye = ye0.copy()
    if valid.sum() > 100:
        xs = np.arange(W)
        ye_i = np.interp(xs, xs[valid], ye0[valid])
        ye_s = cv2.GaussianBlur(ndi.median_filter(ye_i, 101)[None].astype(np.float32), (0, 0), 14)[0]
        yi = np.clip(np.round(ye_s).astype(int), 279, 286)
        spk = np.zeros((H, W), np.uint8)
        nonbrace = (np.arange(W) < 740) | (np.arange(W) > 1290)
        valid = nonbrace
        for x in np.where(valid)[0]:
            spk[yy0 - 10:yi[x], x] = (mx[yy0 - 10:yi[x], x] < 22)
        spk = cv2.dilate(spk, np.ones((3, 3), np.uint8))
        u8 = np.clip(rgb, 0, 255).astype(np.uint8)
        inp = cv2.inpaint(u8, spk * 255, 5, cv2.INPAINT_TELEA).astype(np.float32)
        rgb = rgb.copy()
        for x in np.where(valid)[0]:
            rgb[yy0 - 10:yi[x], x] = inp[yy0 - 10:yi[x], x]
            rgb[yi[x]:yi[x] + 12, x] = 0
        print('top-rail edge cleaned, mean y', ye_s[valid].mean(), 'cols', valid.sum())
    else:
        yi = None; valid = None
    mx = rgb.max(2)
    # metal envelope: close the speckled dark strips along the brace faces (dither noise) so they count as metal, then fill them
    CR = 8
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CR + 1, 2 * CR + 1))
    T = cv2.morphologyEx((mx >= 14).astype(np.uint8), cv2.MORPH_CLOSE, kern) > 0
    black = ~T
    noise = T & (mx < 24)              # dark noise inside the envelope
    good = T & (mx >= 24)
    # restrict the filling to the neighbourhood of the opening (do not touch real dark shadow gaps elsewhere)
    near_open = ndi.binary_dilation(black, iterations=30)
    noise &= near_open
    wg = cv2.GaussianBlur((good & near_open).astype(np.float32), (0, 0), 3.5)
    nm = cv2.GaussianBlur(rgb * (good & near_open)[..., None], (0, 0), 3.5)
    fill = nm / np.maximum(wg, 1e-4)[..., None]
    okf = noise & (wg > 0.03)
    rgb = rgb.copy(); rgb[okf] = fill[okf]
    print('filled noise px', int(okf.sum()))
    mx = rgb.max(2)
    lab_, n = ndi.label(black)
    sizes = ndi.sum(black, lab_, index=np.arange(1, n + 1))
    big = np.where(sizes > 20000)[0] + 1      # all the empty regions (inside triangle and left/right of the braces)
    print('big comps', [(int(i), int(sizes[i-1])) for i in big])
    M = np.isin(lab_, big)
    print('opening px frac', M.mean(), 'labels', n)
    # dark slab between the two brace heads at the apex (shadow, values 17-48) belongs to the opening
    win = np.zeros_like(M); win[264:306, 1016:1044] = True
    cand = (mx < 36) & win
    l3, n3 = ndi.label(M | cand)
    M = M | (cand & np.isin(l3, np.unique(l3[M])))
    M = ndi.binary_fill_holes(M) | M
    if yi is not None:
        for x in np.where(valid)[0]:
            M[yy0 - 10:300, x] = np.arange(yy0 - 10, 300) >= yi[x]
    # smooth the ragged threshold boundary (baked soft shadows give a noisy <14 contour)
    Ms = cv2.GaussianBlur(M.astype(np.float32), (0, 0), SM) > 0.5
    # keep only big regions after smoothing (drops noise); fill holes
    lab2, n2 = ndi.label(Ms)
    sz2 = ndi.sum(Ms, lab2, index=np.arange(1, n2 + 1))
    Ms = np.isin(lab2, np.where(sz2 > 20000)[0] + 1)
    M = Ms
    # signed distance: >0 outside opening (metal side), <0 inside
    dout = ndi.distance_transform_edt(~M)
    din = ndi.distance_transform_edt(M)
    sd = np.where(M, -din + 0.5, dout - 0.5)
    BAND = 5
    solid = (~M) & (dout > BAND)
    wgt = cv2.GaussianBlur(solid.astype(np.float32), (0, 0), 3.0)
    num = cv2.GaussianBlur(rgb * solid[..., None], (0, 0), 3.0)
    ref = num / np.maximum(wgt, 1e-4)[..., None]
    idx = ndi.distance_transform_edt(~solid, return_distances=False, return_indices=True)
    nearest = rgb[idx[0], idx[1]]
    ref = np.where((wgt > 0.02)[..., None], ref, nearest)
    # alpha: linear ramp over AA px centred on the contour
    AA = AA_W
    alpha = np.clip((sd - SHIFT) / AA + 0.5, 0, 1)
    alpha[M] = 0
    # colour: nearest-solid colour in the edge band, original texture beyond
    k = np.clip((sd - 2.5) / 2.5, 0, 1)[..., None]           # 0 at <=2.5px, 1 at >=5px
    col = k * rgb + (1 - k) * ref
    # lift the dark baked shadow strip along the brace sides (reads as a black outline on light backgrounds)
    deep = (~M) & (dout >= 20) & (dout <= 34)
    wd = cv2.GaussianBlur(deep.astype(np.float32), (0, 0), 7)
    nd = cv2.GaussianBlur(rgb * deep[..., None], (0, 0), 7)
    ref_deep = nd / np.maximum(wd, 1e-4)[..., None]
    wlift = np.clip((16.0 - dout) / 8.0, 0, 1) * (wd > 0.02)
    yy = np.arange(H)[:, None]
    wlift = wlift * ((yy > 300) & (yy < 1740))
    lifted = np.maximum(col, LIFT * ref_deep)
    col = col * (1 - wlift[..., None]) + lifted * wlift[..., None]
    col[sd > BAND] = rgb[sd > BAND]
    col[M] = 0
    rgba = np.dstack([np.clip(col, 0, 255), alpha * 255]).round().astype(np.uint8)
    rgba[..., :3][rgba[..., 3] == 0] = 0
    Image.fromarray(rgba, 'RGBA').save('output.png')
    print('coverage', (rgba[..., 3] / 255).mean())
