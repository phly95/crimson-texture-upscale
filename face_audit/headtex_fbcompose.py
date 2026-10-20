"""headtex_fbcompose.py BASESEED PORTSEED: fbhead1 cartoon texture (1024) = own-layout cartoon redraw (cap, ear, collar) + face baked from the symmetric front cartoon portrait
(front-facing triangles only, weighted by how squarely they face the camera) + vector round glasses drawn in the front view + collapsed right-half triangles filled with the
mirror image of the left half + the neck-front strip that wraps to the top rows -> headtex/fbhead1_c_1024.png"""
import sys, json
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_render as HR, headtex_triid as TI, headtex_decal as DC, headtex_fbface as FP
D = FP.D; MODEL, PACK = FP.MODEL, FP.PACK; S = FP.S; OX, OY = FP.OFF; CW = FP.CW; N = 1024
GL = json.load(open(D / "fb_glasses.json")) if (D / "fb_glasses.json").exists() else None      # glasses in portrait-crop px: {"cx","cy","r","w","bridge_y"} (left lens; right lens mirrored about CW/2)
KEYS = [(0, -30), (56, 29), (440, 540), (506.7, 599), (560, 653), (717, 788), (927, 993), (1024, 1090)]       # portrait y -> mesh-view y (cap top, eyes, mouth, chin)
SX = 0.88                                                                                       # portrait -> mesh-view x scale about the centre line
GLASS_P = (386.7, 506.7, 88.4)                                                                  # klein's left lens (circle fit on the raw portrait s3)
def ymap():
    py = np.arange(-200, 1300, dtype=np.float64); my = np.interp(py, *zip(*KEYS), left=np.nan, right=np.nan)
    a, b = KEYS[0], KEYS[-1]; my = np.where(py < a[0], a[1] + (py - a[0]), np.where(py > b[0], b[1] + (py - b[0]), my)); my = cv2.GaussianBlur(my.reshape(1, -1), (0, 0), 25).ravel()
    return py, my
def smooth_skin(w, gm):
    """no wrinkles: morphological closing (removes thin dark lines) on the forehead band and around the eyes outside the lashes, only on skin pixels"""
    cx, cy, r = gm; yy, xx = np.mgrid[0:CW, 0:CW]; u8 = np.clip(w, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(int); skin = (lab[..., 1] > 136) & (lab[..., 0] > 120)
    zone = np.zeros((CW, CW), bool); zone |= (yy > FOREHEAD[0]) & (yy < FOREHEAD[1])
    for c in (cx, CW - cx): d = np.hypot(xx - c, yy - cy); zone |= (d > r * 0.55) & (d < r * 1.25) & (yy > cy + r * 0.2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)); cl = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k); cl = cv2.GaussianBlur(cl, (0, 0), 1.2)
    m = cv2.erode((zone & skin).astype(np.uint8), np.ones((3, 3), np.uint8)); a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3)[..., None]
    return w * (1 - a) + cl.astype(np.float32) * a
FOREHEAD = (330, 530)
def deghost(w, gm):
    """the erased klein rim leaves a faint lighter ring (lens tint edge): smooth the skin in a band around it (normalised blur over skin pixels only)"""
    cx, cy, r = gm; yy, xx = np.mgrid[0:CW, 0:CW]; u8 = np.clip(w, 0, 255).astype(np.uint8); lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(int)
    skin = ((lab[..., 1] > 136) & (lab[..., 0] > 120)).astype(np.float32)[..., None]; out = w.copy()
    for c in (cx, CW - cx):
        d = np.hypot(xx - c, yy - cy); zone = ((d > r * 0.80) & (d < r * 1.22)).astype(np.float32)[..., None] * skin
        bl = cv2.GaussianBlur(w * skin, (0, 0), 7) / np.maximum(cv2.GaussianBlur(skin, (0, 0), 7), 1e-3)[..., None]
        a = cv2.GaussianBlur(zone, (0, 0), 2)[..., None] if zone.ndim == 2 else cv2.GaussianBlur(zone[..., 0], (0, 0), 2)[..., None]; out = out * (1 - a) + bl * a
    return out
def holefill(w, raw, ref):
    """mesh-view areas (non-background in the render `ref`) that the redraw `w` leaves as background (or as its outer outline) get the nearest inner redraw colour, softened"""
    bgc = np.median(np.r_[raw[:20, :20].reshape(-1, 3), raw[:20, -20:].reshape(-1, 3)], 0); nb = (np.abs(w.astype(int) - bgc).sum(-1) <= 30).astype(np.uint8)
    n_, lbl = cv2.connectedComponents(nb, connectivity=4); edge = set(np.unique(np.r_[lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]).tolist()) - {0}
    fg = ~np.isin(lbl, list(edge))                                                                  # background = bg-coloured pixels connected to the border (dark lash lines inside stay)
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); fg = cv2.erode(fg, np.ones((15, 15), np.uint8))   # drop the anti-aliased outline into the bg
    mesh = (np.abs(ref.astype(int) - 40).sum(-1) > 6).astype(np.uint8); hole = (cv2.dilate(mesh, np.ones((9, 9), np.uint8)) > 0) & (fg == 0)
    if hole.any():
        fg = cv2.erode(fg, np.ones((17, 17), np.uint8))                                          # source: inside the drawn outline (no dark contour smeared into the hole)
        hole = hole | ((cv2.dilate(mesh, np.ones((9, 9), np.uint8)) > 0) & (fg == 0) & (cv2.dilate(fg, np.ones((25, 25), np.uint8)) == 0))
        _, lab = cv2.distanceTransformWithLabels((fg == 0).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL); ys, xs = np.nonzero(fg); idx = np.zeros(lab.max() + 1, int)
        idx[lab[fg > 0]] = np.arange(len(ys)); near = w[ys[idx[lab]], xs[idx[lab]]]; fill = cv2.GaussianBlur(near, (0, 0), 6)
        w = np.where(hole[..., None], fill, w)
    return w
def portrait(seed, glass=GLASS_P):
    """raw klein portrait: erase its glasses, map into the mesh view (x scale about the centre, smooth monotone y map through eyes/mouth/chin), mirror the left half,
    fill the mesh area the portrait does not cover (hexagonal cap corners) from the nearest portrait content -> float RGB (CW x CW), mesh-view glasses centre/radius"""
    b = np.array(Image.open(D / f"fbp_s{seed}.png").convert("RGB").resize((CW, CW), Image.LANCZOS)); g = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(int); hsv = cv2.cvtColor(b, cv2.COLOR_RGB2HSV)
    dark = ((g < 110) & (hsv[..., 1] < 110)).astype(np.uint8); yy, xx = np.mgrid[0:CW, 0:CW]; gx, gy, R = glass; m = np.zeros((CW, CW), np.uint8)
    for cx in (gx, 2 * 512 - gx + 0.0, 631.8): m |= (np.abs(np.hypot(xx - cx, yy - gy) - R) < 13).astype(np.uint8)
    m |= ((yy > gy - 50) & (yy < gy + 20) & (np.abs(xx - 512) < 512 - gx + R + 200) & (np.hypot(xx - gx, yy - gy) > R - 6) & (np.hypot(xx - (1024 - gx), yy - gy) > R - 6)).astype(np.uint8)
    arm = (np.abs(yy - gy) < 28) & ((xx < gx - R + 4) | (xx > CW - gx + R - 4)) & (cv2.cvtColor(b, cv2.COLOR_RGB2LAB)[..., 0] < 150)   # klein's temple arms, any dark colour
    era = cv2.dilate((m & dark) | arm.astype(np.uint8), np.ones((5, 5), np.uint8)); b = cv2.inpaint(b, era, 7, cv2.INPAINT_TELEA)
    py, my = ymap(); inv_y = np.interp(np.arange(CW), my, py); mx = (512 + (np.arange(CW) - 512) / SX)
    Mx, My = np.meshgrid(mx.astype(np.float32), inv_y.astype(np.float32)); w = cv2.remap(b, Mx, My, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    h = CW // 2; w[:, h:] = w[:, :h][:, ::-1]
    w = holefill(w, b, np.array(Image.open(D / "fbp_ref.png").convert("RGB")))
    gm = (512 + (gx - 512) * SX, float(np.interp(gy, py, my)), R * 0.9)
    w = smooth_skin(w, gm); w = deghost(w, gm)
    return w.astype(np.float32), gm
def glasses(img, g, erase=True, skin=None):
    """vector round wire glasses on the symmetric portrait (crop px): erase the drawn rims (annulus inpaint), then two perfect circles + bridge + arms"""
    H, W = img.shape[:2]; K = 4; cx2 = CW - g["cx"]; out = img.astype(np.uint8)
    if erase:
        m = np.zeros((H, W), np.uint8)
        for cx in (g["cx"], cx2): cv2.circle(m, (int(cx), int(g["cy"])), int(g["r"] + g.get("er", 14)), 255, -1); cv2.circle(m, (int(cx), int(g["cy"])), int(max(g["r"] - g.get("er", 14), 1)), 0, -1)
        cv2.rectangle(m, (int(g["cx"] + g["r"]), int(g["bridge_y"] - 14)), (int(cx2 - g["r"]), int(g["bridge_y"] + 14)), 255, -1)
        out = cv2.inpaint(out, m, 9, cv2.INPAINT_TELEA)
    big = np.zeros((H * K, W * K), np.uint8); col = np.array(g.get("col", (52, 38, 34)), np.float32); w = int(g["w"] * K)
    for cx in (g["cx"], cx2): cv2.circle(big, (int(cx * K), int(g["cy"] * K)), int(g["r"] * K), 255, w, cv2.LINE_AA)
    # bridge: a gentle arc between the inner rims; arms: straight from the outer rims towards the ears
    xs = np.linspace(g["cx"] + g["r"] * 0.94, cx2 - g["r"] * 0.94, 40); ys = g["bridge_y"] - g.get("arch", 6) * np.sin(np.pi * (xs - xs[0]) / (xs[-1] - xs[0]))
    cv2.polylines(big, [np.round(np.c_[xs, ys] * K).astype(np.int32)], False, 255, w, cv2.LINE_AA)
    armb = np.zeros((H * K, W * K), np.uint8)
    for cx, sgn in ((g["cx"], -1), (cx2, 1)):
        x0 = cx + sgn * g["r"]; cv2.line(armb, (int(x0 * K), int(g["cy"] * K - g["r"] * 0.15 * K)), (int((x0 + sgn * g.get("arm", 60)) * K), int((g["cy"] - g["r"] * 0.3) * K)), 255, w, cv2.LINE_AA)
    if skin is None: lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(int); skin = (lab[..., 1] > 136) & (lab[..., 0] > 110)
    skin = cv2.erode(skin.astype(np.uint8), np.ones((5, 5), np.uint8))   # arms only over skin: they vanish under the woolly hood
    a = (cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) + cv2.resize(armb, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) * cv2.GaussianBlur(skin.astype(np.float32), (0, 0), 1.0)).clip(0, 255)[..., None] / 255
    hl = np.zeros((H * K, W * K), np.uint8)                                                         # thin light glint on the upper-left of each rim
    for cx in (g["cx"], cx2): cv2.ellipse(hl, (int(cx * K), int(g["cy"] * K)), (int((g["r"] - g["w"] * 0.2) * K),) * 2, 0, 200, 250, 255, max(1, int(g["w"] * 0.35 * K)), cv2.LINE_AA)
    ha = cv2.resize(hl, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255 * 0.55
    out = out.astype(np.float32) * (1 - a) + col * a; out = out * (1 - ha) + np.array((170, 160, 150), np.float32) * ha
    return out
def face_tris():
    V, T = HR.tris(MODEL, PACK); ids = TI.idmap(MODEL, PACK, 180, S=S); bad = set(HR.broken(V, T, 0.2)); X, Y, Z, _ = TI.proj(MODEL, PACK, 180, 0, S)
    vis = [j for j in np.unique(ids[ids >= 0]) if j not in bad]; out = []
    cen = V.mean(0); a = np.radians(180)
    for j in vis:
        vi = list(T[j][0]); P = V[vi]; n = np.cross(P[1] - P[0], P[2] - P[0]); n /= np.linalg.norm(n)
        if n @ (P.mean(0) - cen) < 0: n = -n                                                          # outward normal
        nz = -np.sin(a) * n[0] + np.cos(a) * n[2]; cx = X[vi].mean()                                 # > 0: faces the camera (larger Z is nearer)
        if nz > 0 and cx < S / 2 + 2 and j not in (18, 19, 20, 21): out.append((j, nz))             # textured (left) half, no neck
    return out
def facing(yaw, S_):
    """per triangle: outward-normal facing towards the camera of view `yaw` (>0 = front-facing) and visible fraction of its projected area"""
    V, T = HR.tris(MODEL, PACK); X, Y, Z, _ = TI.proj(MODEL, PACK, yaw, 0, S_); ids = TI.idmap(MODEL, PACK, yaw, S=S_); cnt = np.bincount(ids[ids >= 0].ravel(), minlength=len(T))
    cen = V.mean(0); a = np.radians(yaw); f, v = np.zeros(len(T)), np.zeros(len(T))
    for j, (vi, uv, nm) in enumerate(T):
        vi = list(vi); P = V[vi]; n = np.cross(P[1] - P[0], P[2] - P[0]); n /= max(np.linalg.norm(n), 1e-12)
        if n @ (P.mean(0) - cen) < 0: n = -n
        f[j] = -np.sin(a) * n[0] + np.cos(a) * n[2]; ar = abs((X[vi[1]] - X[vi[0]]) * (Y[vi[2]] - Y[vi[0]]) - (X[vi[2]] - X[vi[0]]) * (Y[vi[1]] - Y[vi[0]])) / 2; v[j] = cnt[j] / max(ar, 1)
    return f, v
NECK = (18, 19, 20, 21); SIDE = "sideL"
def clean_base(base, wool_lab):
    """the own-layout redraw's grey wool and cyan glow halos -> the cream wool of the portrait (lightness shading kept, chroma replaced)"""
    lab = cv2.cvtColor(np.clip(base, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32); a, b = lab[..., 1] - 128, lab[..., 2] - 128
    wool = ((np.hypot(a, b) < 14) & (lab[..., 0] > 110)) | ((b < 2) & (lab[..., 0] > 90)); w = cv2.GaussianBlur(wool.astype(np.float32), (0, 0), 2)[..., None]
    t = lab.copy(); t[..., 0] = lab[..., 0] * (wool_lab[0] / max(np.median(lab[wool][:, 0]), 1)); t[..., 1:] = wool_lab[1:]
    return cv2.cvtColor(np.clip(lab * (1 - w) + t * w, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
def paint_neck(res, skin_rgb, collar_rgb):
    """neck box: front quad (18-21, its lower third wraps to the top rows) plain neck skin with a soft shadow under the chin; back/side quads (46-49) plain brown collar"""
    N = res.shape[0]; V, T = HR.tris(MODEL, PACK); H = N + N // 8; tall = np.zeros((H, N), np.uint8); col = np.zeros((H, N), np.uint8)
    for j, c in ((18, 1), (19, 1), (20, 1), (21, 1), (46, 2), (47, 2), (48, 2), (49, 2)):
        u = np.round(np.array(T[j][1]) * [N, N] * 16).astype(np.int32); cv2.fillPoly(col, [u], c, shift=4); cv2.polylines(col, [u], True, c, max(5, N // 200), shift=4)
    out = np.vstack([res, res[:N // 8]]).astype(np.float32); vv = np.arange(H, dtype=np.float32)[:, None, None] / N
    skin = np.array(skin_rgb, np.float32) * np.clip(0.86 + (vv - 0.94) * 1.6, 0.86, 1.0); brown = np.array(collar_rgb, np.float32) * np.clip(0.92 + (vv - 0.8) * 0.5, 0.85, 1.05)
    for c, fill in ((1, skin), (2, brown)):
        a = cv2.GaussianBlur((col == c).astype(np.float32), (0, 0), 1.5 * N / 1024)[..., None]; out = out * (1 - a) + np.broadcast_to(fill, out.shape) * a
    res = out[:N].copy(); wrap = col[N:] > 0; res[:N // 8][wrap] = out[N:][wrap]                                         # v > 1 wraps (the engine repeats this texture vertically)
    return res
def band_fill(res):
    """collapsed right-half triangles (36, 37 = right eye/cheek quad, UVs on a line): they cannot hold an eye, so their texel band gets a smooth continuation of the skin around it"""
    N = res.shape[0]; k = N / 1024; V, T = HR.tris(MODEL, PACK); m = np.zeros((N, N), np.uint8)
    for j in HR.broken(V, T, 0.1):
        u = np.round(np.array(T[j][1]) * [N, N] * 16).astype(np.int32); cv2.fillPoly(m, [u], 255, shift=4); cv2.polylines(m, [u], True, 255, max(3, int(3 * k)), shift=4)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8)); res8 = np.clip(res, 0, 255).astype(np.uint8); lab = cv2.cvtColor(res8, cv2.COLOR_RGB2LAB).astype(int)
    sk = ((lab[..., 0] > 140) & (lab[..., 1] > 136) & (m == 0)).astype(np.float32)[..., None]                                # skin texels around the band (no rim, no brow)
    fill = cv2.GaussianBlur(res8.astype(np.float32) * sk, (0, 0), 8 * k) / np.maximum(cv2.GaussianBlur(sk, (0, 0), 8 * k)[..., None], 1e-3)
    a = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 1.0 * k)[..., None]; return res8 * (1 - a) + fill * a
def glasses_layer(P, gl, K):
    """the vector glasses alone as (colour, alpha) at K x the crop resolution (arms masked by the skin of P)"""
    global CW
    Pk = cv2.resize(np.clip(P, 0, 255).astype(np.uint8), (CW * K, CW * K), interpolation=cv2.INTER_CUBIC).astype(np.float32); g = {k_: (v * K if k_ in ("cx", "cy", "r", "w", "bridge_y", "arch", "arm") else v) for k_, v in gl.items()}
    cw0 = CW; CW = cw0 * K
    try:
        lab = cv2.cvtColor(Pk.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(int); skin = ((lab[..., 1] > 136) & (lab[..., 0] > 110))
        o0 = glasses(np.where(skin[..., None], np.float32(0), np.float32(0)) + 0 * Pk, g, erase=False, skin=skin); o1 = glasses(np.full_like(Pk, 255), g, erase=False, skin=skin)
    finally: CW = cw0
    A = np.clip(1 - (o1 - o0)[..., :1] / 255, 0, 1); C = o0 / np.maximum(A, 1e-3); return C, A[..., 0]
def finish(img, out_name="fbhead1_c3_2048.png"):
    """after the 2x detail pass: re-bake the vector glasses at full resolution through the front-facing triangles, re-smooth the collapsed band, repaint the neck"""
    prm = json.load(open(D / "fbhead1_c2_params.json")); gl = prm["gl"]; N2 = img.shape[0]; K = N2 // N
    P, _ = portrait(prm["pseed"]); C, A = glasses_layer(P, gl, K); lay = np.dstack([C, A]).astype(np.float32)
    ft = dict(face_tris()); out = img.astype(np.float32); done = np.zeros(img.shape[:2], bool)
    for j, f in sorted(ft.items(), key=lambda t: -t[1]):
        if f <= 0.5: continue
        sx, sy = DC.screen_of(MODEL, PACK, 180, S * K, [j], N2, N2, grow=2); ok = np.isfinite(sx) & ~done
        c = cv2.remap(lay, np.where(ok, sx - OX * K - .5, -10).astype(np.float32), np.where(ok, sy - OY * K - .5, -10).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        a = (c[..., 3] * ok)[..., None]; out = out * (1 - a) + c[..., :3] * a; done |= ok
    out = band_fill(out); out = paint_neck(out, np.array(prm["skin"]), np.array(prm["collar"]))
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(D / out_name); return out
def compose(bseed, pseed, side=None, back=None, dump=True, out_name="fbhead1_c2_1024.png"):
    base = np.array(Image.open(D / f"fbhead1_s{bseed}_fin.png").convert("RGB").resize((N, N), Image.LANCZOS)).astype(np.float32)
    P, (gcx, gcy, gr) = portrait(pseed); gcx += 8; gr *= 0.88; gl = dict(cx=gcx, cy=gcy, r=gr, w=8.5, bridge_y=gcy - 0.25 * gr, arch=8, arm=70)
    P = glasses(P, gl, erase=False); Image.fromarray(np.clip(P, 0, 255).astype(np.uint8)).save(D / f"fbp_s{pseed}_mapped.png")
    import headtex_fbviews as FV
    views = [("front", 180, S, (OX, OY), P)]
    for nm, sd in ((SIDE, side), ("back", back)):
        if sd is None: continue
        y, s_, off, _ = FV.VIEWS[nm]; raw = np.array(Image.open(D / f"fbv_{nm}_s{sd}_fin.png").convert("RGB")); hf = holefill(raw, raw, np.array(Image.open(D / f"fbv_{nm}_ref.png").convert("RGB")))
        Image.fromarray(hf).save(D / f"fbv_{nm}_s{sd}_fill.png"); views.append((nm, y, s_, off, hf.astype(np.float32)))
    V, T = HR.tris(MODEL, PACK); bad = set(HR.broken(V, T, 0.1)); ft = dict(face_tris()); fv = {v[0]: facing(v[1], v[2]) for v in views}
    plan = []
    for j in range(len(T)):
        if j in bad or j in NECK: continue
        if j in ft and ft[j] > 0.5: plan.append((ft[j] + 1, j, 0)); continue                       # face: always the front portrait
        best = max(((fv[v[0]][0][j], k) for k, v in enumerate(views) if k > 0 and fv[v[0]][1][j] > 0.7), default=(0, -1))
        if best[1] > 0 and best[0] > 0.25: plan.append((best[0], j, best[1]))
    wl = cv2.cvtColor(np.clip(views[-1][4], 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32); wl = np.median(wl[(wl[:, 0] > 200)], 0)
    base = clean_base(base, wl)
    out = base.copy(); wsum = np.zeros((N, N), np.float32); src = np.full((N, N), -1, np.int8)
    for f, j, k in sorted(plan, reverse=True):                                                    # most squarely seen first; a texel is written once
        nm, yaw, s_, (ox, oy), img = views[k]
        sx, sy = DC.screen_of(MODEL, PACK, yaw, s_, [j], N, N, grow=2); ok = np.isfinite(sx) & (wsum < 0.5)
        mx, my = np.where(ok, sx - ox - .5, -10).astype(np.float32), np.where(ok, sy - oy - .5, -10).astype(np.float32)
        lay = np.dstack([img, np.ones(img.shape[:2], np.float32)])
        c = cv2.remap(lay, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        kk = ok & (c[..., 3] > 0.99); out[kk] = c[..., :3][kk]; wsum[kk] = 1; src[kk] = k
    wb = cv2.GaussianBlur(wsum, (0, 0), 3)[..., None]
    res = base * (1 - wb) + out * wb
    if dump: Image.fromarray((src + 1).astype(np.uint8) * 80).save(D / "fbhead1_c_src.png")
    res = band_fill(res)
    pl = cv2.cvtColor(np.clip(P, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB); skin_rgb = np.median(P[700:840, 380:640].reshape(-1, 3), 0)          # cheek/chin skin of the portrait
    bk = views[-1][4]; collar_rgb = np.median(bk[930:1010, 300:720].reshape(-1, 3), 0); print("neck skin", skin_rgb.round(), "collar", collar_rgb.round())
    res = paint_neck(res, skin_rgb, collar_rgb); json.dump(dict(gl=gl, skin=skin_rgb.tolist(), collar=collar_rgb.tolist(), pseed=pseed), open(D / "fbhead1_c2_params.json", "w"))
    if dump: Image.fromarray(np.clip(res, 0, 255).astype(np.uint8)).save(D / out_name)
    return res
def compose_hi(bseed, pseed, side, back, K=2, out_name="fbhead1_c4_2048.png"):
    """same bake as compose(), at K x resolution, from the K x detail-passed SOURCE views (fbp_s<p>_mapped_x2, fbv_*_fill_x2; a detail pass on the atlas itself turns the
    triangle seams into drawn lines), then the vector glasses re-baked at full resolution, collapsed band smoothed, neck painted"""
    import headtex_fbviews as FV
    N2 = N * K; L = lambda f: np.array(Image.open(D / f).convert("RGB").resize((CW * K, CW * K), Image.LANCZOS)).astype(np.float32)
    views = [("front", 180, S * K, (OX * K, OY * K), L(f"fbp_s{pseed}_mapped_x2.png"))]
    for nm, sd in ((SIDE, side), ("back", back)):
        y, s_, (ox, oy), _ = FV.VIEWS[nm]; views.append((nm, y, s_ * K, (ox * K, oy * K), L(f"fbv_{nm}_s{sd}_fill_x2.png")))
    V, T = HR.tris(MODEL, PACK); bad = set(HR.broken(V, T, 0.1)); ft = dict(face_tris()); fv = {v[0]: facing(v[1], v[2] // K) for v in views}
    plan = []
    for j in range(len(T)):
        if j in bad or j in NECK: continue
        if j in ft and ft[j] > 0.5: plan.append((ft[j] + 1, j, 0)); continue
        best = max(((fv[v[0]][0][j], k) for k, v in enumerate(views) if k > 0 and fv[v[0]][1][j] > 0.7), default=(0, -1))
        if best[1] > 0 and best[0] > 0.25: plan.append((best[0], j, best[1]))
    wl = cv2.cvtColor(np.clip(views[-1][4], 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32); wl = np.median(wl[(wl[:, 0] > 200)], 0)
    base = clean_base(np.array(Image.open(D / f"fbhead1_s{bseed}_fin.png").convert("RGB").resize((N2, N2), Image.LANCZOS)).astype(np.float32), wl)
    out = base.copy(); wsum = np.zeros((N2, N2), np.float32)
    for f, j, k in sorted(plan, reverse=True):
        nm, yaw, s_, (ox, oy), img = views[k]
        sx, sy = DC.screen_of(MODEL, PACK, yaw, s_, [j], N2, N2, grow=2 * K); ok = np.isfinite(sx) & (wsum < 0.5)
        lay = np.dstack([img, np.ones(img.shape[:2], np.float32)])
        c = cv2.remap(lay, np.where(ok, sx - ox - .5, -10).astype(np.float32), np.where(ok, sy - oy - .5, -10).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        kk = ok & (c[..., 3] > 0.99); out[kk] = c[..., :3][kk]; wsum[kk] = 1
    wb = cv2.GaussianBlur(wsum, (0, 0), 3 * K)[..., None]; res = base * (1 - wb) + out * wb
    Image.fromarray(np.clip(res, 0, 255).astype(np.uint8)).save(D / "fbhead1_c4_prefinish.png")
    return finish(res, out_name=out_name)
if __name__ == "__main__":
    if sys.argv[1] == "hi": compose_hi(*[int(x) for x in sys.argv[2:6]]); sys.exit()
    if sys.argv[1] == "finish": finish(np.array(Image.open(sys.argv[2]).convert("RGB"))); sys.exit()
    a = [None if x == "-" else int(x) for x in sys.argv[1:5]] + [None] * 4; compose(a[0], a[1], a[2], a[3], out_name=sys.argv[5] if len(sys.argv) > 5 else "fbhead1_c2_1024.png")
