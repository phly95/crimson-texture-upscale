"""dirt_layer.py - procedural dirt / weathering layer for textures.

    from dirt_layer import add_dirt
    out = add_dirt(img, strength=0.5, direction="down", seed=1)      # img: HxWx3 or HxWx4 uint8 array -> same shape; alpha is kept untouched

What it paints (all procedural, deterministic for a seed, sizes relative to the picture so it works at any resolution):
  (copy_dirt: the original texture's own dirt pattern, carried over to the redraw)
  grime    mottled multi-scale noise, darkening toward a dirty brown
  edges    grime gathering in recesses: panel lines, rivet rows, creases and silhouettes (found from the picture's own edge map)
  streaks  long thin run-off streaks (rain, oil) along `direction` (down | up | left | right | none)
  soot     dark exhaust-like smoke staining that fades along `direction`
  dust     a pale dusty film, thicker in patches
  wear     fine bright scratches and edge chips (bare metal showing through)
CLI (used by the review page's Add dirt button):
    python dirt_layer.py options UID VERSION    -> previews units/UID/dirt/<version>/opt_<id>.jpg (split dark/light for cut-outs), prints JSON [{id,label}]
    python dirt_layer.py apply UID OPTION VERSION  -> saves the result as a NEW version (custom:N, label "Dirt: ...") in units/UID/custom/ (the clean one is never overwritten)
"""
import sys, os, json, re, time
from pathlib import Path
import numpy as np
import cv2


def _noise(h, w, rng, cells, octaves=5, persist=0.55):
    """fractal value noise 0..1; `cells` = grid cells across the shorter side for the first octave"""
    s = min(h, w)
    out = np.zeros((h, w), np.float32); amp, tot = 1.0, 0.0
    for o in range(octaves):
        c = cells * (2 ** o)
        gh, gw = max(2, int(round(c * h / s))), max(2, int(round(c * w / s)))
        g = rng.random((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp; amp *= persist
    out /= tot
    lo, hi = np.percentile(out, [2, 98])
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)


def _streaks(h, w, rng, direction, length_frac=0.45, density=1.0):
    """thin run-off streaks along `direction`; 0..1"""
    s = min(h, w)
    vertical = direction in ("down", "up")
    n = rng.random((h, w)).astype(np.float32)
    n = (n > (1 - 0.10 * density)).astype(np.float32) * rng.random((h, w)).astype(np.float32)      # sparse random seeds
    L = max(5, int((h if vertical else w) * length_frac))
    k = np.zeros((L, 1) if vertical else (1, L), np.float32)
    ramp = np.linspace(1, 0, L, dtype=np.float32) ** 1.5                                          # a streak starts strong and fades
    if direction in ("up", "left"):
        ramp = ramp[::-1]
    k[:, 0] = ramp if vertical else 0
    if not vertical:
        k[0, :] = ramp
    n = cv2.filter2D(n, -1, k / k.sum() * 3.0, borderType=cv2.BORDER_REFLECT)
    n = cv2.GaussianBlur(n, (0, 0), max(0.6, s * 0.0016)) if vertical else cv2.GaussianBlur(n, (0, 0), max(0.6, s * 0.0016))
    n *= (0.5 + 0.5 * _noise(h, w, rng, 3, 3))                                                     # streaks come and go along their length
    hi = np.percentile(n, 99.5) + 1e-6
    return np.clip(n / hi, 0, 1)


def _edge_map(rgb, s):
    """0..1: where the picture itself has lines and creases (panel seams, rivets, outlines), smeared a little so grime collects beside them"""
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lum = cv2.GaussianBlur(lum, (0, 0), max(0.7, s * 0.0015))
    gx, gy = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), max(1.0, s * 0.004))
    hi = np.percentile(mag, 97) + 1e-6
    return np.clip(mag / hi, 0, 1) ** 0.8


def add_dirt(img, strength=0.5, grime=1.0, edges=1.0, streaks=0.5, direction="down", soot=0.0, dust=0.0, wear=0.0,
             tint=(78, 58, 38), seed=1):
    a = np.asarray(img)
    has_a = a.ndim == 3 and a.shape[2] == 4
    rgb = a[..., :3].astype(np.uint8)
    h, w = rgb.shape[:2]; s = min(h, w)
    rng = np.random.default_rng(seed)
    f = rgb.astype(np.float32)
    D = np.zeros((h, w), np.float32)
    if grime > 0:
        D += grime * 0.9 * (_noise(h, w, rng, 3, 5) ** 1.8) * (0.25 + 0.75 * _noise(h, w, rng, 9, 3))                       # patchy: some areas stay clean
    if edges > 0:
        D += edges * 0.9 * _edge_map(rgb, s) * (0.45 + 0.55 * _noise(h, w, rng, 6, 4))
    if streaks > 0 and direction != "none":
        D += streaks * 1.1 * _streaks(h, w, rng, direction)
    D = np.clip(D, 0, 1.6)
    D = np.clip((D - 0.12) * 1.5, 0, 1) * strength                                              # keep clean areas clean; contrast so it reads as dirt, not as a tint
    t = np.array(tint, np.float32) / 255.0
    dirty = f * (t * 1.55 + 0.15)[None, None, :]                                                    # multiply toward brown: darkens and warms
    f = f * (1 - D[..., None]) + dirty * D[..., None]
    if soot > 0:
        ramp = np.linspace(1, 0, w if direction in ("left", "right") else h, dtype=np.float32)
        if direction in ("up", "left"):
            ramp = ramp[::-1]
        ramp = np.tile(ramp[None, :], (h, 1)) if direction in ("left", "right") else np.tile(ramp[:, None], (1, w))
        plume = np.clip(ramp ** 1.2 * (0.35 + 0.9 * _noise(h, w, rng, 4, 4)) * (0.5 + 0.5 * _noise(h, w, rng, 12, 3)), 0, 1) * soot * strength * 1.4
        f = f * (1 - np.clip(plume, 0, 0.9)[..., None] * 0.85)
    if dust > 0:
        m = np.clip((_noise(h, w, rng, 5, 5) - 0.35) * 1.8, 0, 1) * dust
        f = f * (1 - m[..., None] * 0.55) + np.array([205, 190, 160], np.float32) * (m[..., None] * 0.55)
    if wear > 0:
        sc = np.zeros((h, w), np.float32)
        for _ in range(int(25 + 70 * wear * (h * w) ** 0.5 / 512)):
            x0, y0 = rng.integers(0, w), rng.integers(0, h)
            ang = rng.uniform(0, np.pi); ln = rng.uniform(0.02, 0.12) * s
            cv2.line(sc, (int(x0), int(y0)), (int(x0 + np.cos(ang) * ln), int(y0 + np.sin(ang) * ln)), float(rng.uniform(0.4, 1.0)), 1, cv2.LINE_AA)
        sc = cv2.GaussianBlur(sc, (0, 0), max(0.5, s * 0.0008))
        chips = (_edge_map(rgb, s) * _noise(h, w, rng, 14, 3) > 0.62).astype(np.float32)
        chips = cv2.GaussianBlur(chips, (0, 0), max(0.6, s * 0.0012))
        m = np.clip(sc * 0.9 + chips * 0.8, 0, 1) * wear
        f = f * (1 - m[..., None]) + np.clip(f * 1.5 + 45, 0, 255) * m[..., None]
    out = np.clip(f + 0.5, 0, 255).astype(np.uint8)
    return np.dstack([out, a[..., 3]]) if has_a else out


def copy_dirt(img, orig_path, strength=1.0, mode="dark", chroma=0.7, smooth=1.6, extra=None, direction="down", streaks=0.5, tint=(78, 58, 38), seed=1):
    """copy the ORIGINAL texture's own dirt onto the redraw.  The redraw is shrunk to the original's size; what the original has and the redraw lacks
    (blotches, grime patches, darker streaks, colour stains) is the dirt: that residual is smoothed, enlarged and added back to the redraw in the same places.
      mode "dark": only where the original is darker than the redraw (grime)   "both": also lighter patches (dust, bleaching)
      chroma: how much of the original's colour staining is copied      smooth: blur (in original pixels) so pixel noise and misaligned edges do not become dirt
      extra: optional dict of add_dirt() arguments applied on top (more grime in the same style)"""
    from PIL import Image
    a = np.asarray(img); has_a = a.ndim == 3 and a.shape[2] == 4
    rgb = a[..., :3].astype(np.uint8); H, W = rgb.shape[:2]
    o = np.asarray(Image.open(orig_path).convert("RGBA")).astype(np.float32) / 255.0
    oh, ow = o.shape[:2]
    oa = o[..., 3:4]; orgb = np.clip(o[..., :3] / np.maximum(oa, 1e-3), 0, 1) if (oa < 0.99).any() else o[..., :3]
    small = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_AREA)
    lab_o = cv2.cvtColor((orgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_r = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    d = lab_o - lab_r
    d = cv2.GaussianBlur(d, (0, 0), smooth)
    if mode == "map":                                          # the original only says WHERE it is dirty; real dirt texture (mottling, streaks) is painted there
        em = cv2.GaussianBlur(_edge_map(small, min(oh, ow)), (0, 0), 1.0)
        dens = np.maximum(-d[..., 0], 0) * (1.0 - 0.6 * em)
        pos = dens[dens > 0]
        dens = np.clip(dens / (np.percentile(pos, 92) + 1e-6), 0, 1) if pos.size else dens
        dens = cv2.GaussianBlur(cv2.resize(dens, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), max(1.0, 0.45 * W / ow))
        dens = np.clip(dens, 0, 1)
        rng = np.random.default_rng(seed)
        tex = 0.25 + 0.75 * (0.55 * _noise(H, W, rng, 7, 5) + 0.45 * _noise(H, W, rng, 18, 3))
        if streaks > 0 and direction != "none":
            tex = np.clip(tex + streaks * 0.9 * _streaks(H, W, rng, direction), 0, 1.3)
        D = np.clip(dens ** 0.9 * tex * 0.8, 0, 1) * min(strength, 1.6)
        t = np.array(tint, np.float32) / 255.0
        f = rgb.astype(np.float32); dirty = f * (t * 1.55 + 0.15)[None, None, :]
        out = np.clip(f * (1 - D[..., None]) + dirty * D[..., None] + 0.5, 0, 255).astype(np.uint8)
        if extra:
            out = add_dirt(out, **extra)
        return np.dstack([out, a[..., 3]]) if has_a else out
    # the part of the difference that is only edge misalignment: weaker where the redraw has strong edges
    em = cv2.GaussianBlur(_edge_map(small, min(oh, ow)), (0, 0), 1.0)
    d *= (1.0 - 0.75 * em)[..., None]
    if mode == "dark":
        d[..., 0] = np.minimum(d[..., 0], 0)
    d[..., 1:] *= chroma
    d *= strength
    up = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), max(0.8, 0.35 * W / ow))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] = np.clip(lab[..., 0] + up[..., 0], 0, 255)
    lab[..., 1:] = np.clip(lab[..., 1:] + up[..., 1:], 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    if extra:
        out = add_dirt(out, **extra)
    return np.dstack([out, a[..., 3]]) if has_a else out


def fix_edge(img, orig_path, width=0.07, strength=1.0, scratches=1.0, grime=0.6, seed=1):
    """restore the ORIGINAL's outer edge: where the original has a dark, scratched, grimy border (maps, notice boards, worn panels) but the redraw drew a light margin.
    In a band of `width` (fraction of the short side) along the picture edge the redraw's broad tone and colour are pulled back to the original's, keeping its own fine detail,
    then dark scratches and grime are added inside that band (strongest at the very edge)."""
    from PIL import Image
    a = np.asarray(img); has_a = a.ndim == 3 and a.shape[2] == 4
    rgb = a[..., :3].astype(np.uint8); H, W = rgb.shape[:2]; s_ = min(H, W)
    o = np.asarray(Image.open(orig_path).convert("RGB"))
    oh, ow = o.shape[:2]
    ou = cv2.resize(o, (W, H), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[:H, :W]
    dist = np.minimum.reduce([xx, W - 1 - xx, yy, H - 1 - yy]).astype(np.float32)
    wb = np.clip(1.0 - dist / (width * s_), 0, 1) ** 1.2                                   # 1 at the edge, 0 at the inner side of the band
    lab_r = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_o = cv2.cvtColor(ou, cv2.COLOR_RGB2LAB).astype(np.float32)
    sg = max(2.0, 1.6 * W / ow)
    shift = cv2.GaussianBlur(lab_o, (0, 0), sg) - cv2.GaussianBlur(lab_r, (0, 0), sg)
    lab = lab_r + shift * (wb * min(strength, 1.5))[..., None]
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    rng = np.random.default_rng(seed)
    wide = np.clip(1.0 - dist / (width * 1.3 * s_), 0, 1)
    sc = np.zeros((H, W), np.float32)
    for _ in range(int((60 + 260 * scratches) * (H + W) / 2048 * 2)):                        # dark scratches, mostly running along the edge
        side = rng.integers(0, 4); t = rng.random() * (W if side < 2 else H); depth = rng.random() ** 2 * width * 1.3 * s_
        ang = rng.normal(0, 0.5) + (0 if side < 2 else np.pi / 2); ln = rng.uniform(0.02, 0.15) * s_
        x0 = t if side < 2 else (depth if side == 2 else W - 1 - depth); y0 = (depth if side == 0 else H - 1 - depth) if side < 2 else t
        cv2.line(sc, (int(x0), int(y0)), (int(x0 + np.cos(ang) * ln), int(y0 + np.sin(ang) * ln)), float(rng.uniform(0.4, 1.0)), 1, cv2.LINE_AA)
    sc = cv2.GaussianBlur(sc, (0, 0), max(0.5, s_ * 0.0007)) * wide
    gr = (0.4 + 0.6 * _noise(H, W, rng, 14, 4)) * wide * grime
    dark = np.clip(sc * 0.85 * scratches + gr * 0.45, 0, 0.9)
    out = out * (1 - dark[..., None]) + out * 0.25 * dark[..., None]
    res = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return np.dstack([res, a[..., 3]]) if has_a else res


# id, label, kwargs   (kwargs with "orig": copy_dirt from the original texture, otherwise add_dirt)
PRESETS = [
    ("o_map", "Original's dirt pattern, textured grime", dict(orig=dict(strength=1.0, mode="map"))),
    ("o_map_strong", "Original's dirt pattern, heavier", dict(orig=dict(strength=1.5, mode="map", streaks=0.8))),
    ("o_map_flow", "Original's dirt pattern, streaks along the body", dict(orig=dict(strength=1.2, mode="map", direction="right", streaks=1.0))),
    ("o_dark", "Original's dirt, copied directly (soft)", dict(orig=dict(strength=0.7, mode="dark"))),
    ("o_plus", "Original's dirt pattern + extra grime", dict(orig=dict(strength=1.0, mode="map", extra=dict(strength=0.3, grime=0.5, edges=0.6, streaks=0.3, direction="down")))),
    ("edge", "Dark scratched edge (copy the original's border)", dict(edge=dict())),
    ("edge_plain", "Dark edge only, no scratches", dict(edge=dict(scratches=0.0, grime=0.3))),
    ("light", "Light grime", dict(strength=0.4, streaks=0.3, direction="down")),
    ("medium", "Medium grime", dict(strength=0.65, streaks=0.5, direction="down")),
    ("heavy", "Heavy grime", dict(strength=0.95, grime=1.2, streaks=0.7, direction="down")),
    ("rain", "Rain streaks (run down)", dict(strength=0.6, grime=0.35, edges=0.5, streaks=1.3, direction="down")),
    ("flow", "Slipstream streaks (front to back)", dict(strength=0.6, grime=0.4, edges=0.6, streaks=1.3, direction="right")),
    ("seams", "Dirty panel lines and rivets", dict(strength=0.8, grime=0.25, edges=1.6, streaks=0.2, direction="down")),
    ("soot", "Exhaust soot", dict(strength=0.6, grime=0.4, streaks=0.4, direction="right", soot=1.0)),
    ("dust", "Dusty", dict(strength=0.3, grime=0.6, edges=0.5, streaks=0.0, direction="none", dust=0.9)),
    ("worn", "Worn: grime, scratches, chips", dict(strength=0.55, streaks=0.4, direction="down", wear=1.0)),
    ("weathered", "Heavily weathered (everything)", dict(strength=0.85, grime=1.0, edges=1.1, streaks=0.8, direction="down", soot=0.4, dust=0.3, wear=0.7)),
]
LABELS = {p[0]: p[1] for p in PRESETS}
PIPE = Path("/home/deck/workspace/csvm-hd/pipeline")
UN = PIPE / "units"


def render(arr, kw, uid):
    kw = dict(kw)
    if "edge" in kw:
        return fix_edge(arr, str(UN / uid / "orig.png"), **kw["edge"])
    if "orig" in kw:
        return copy_dirt(arr, str(UN / uid / "orig.png"), **kw["orig"])
    return add_dirt(arr, **kw)


def _src(uid, vid):
    d = UN / uid
    m = re.fullmatch(r"(klein|retry):([0-9]{1,3})", vid or "")
    if not m:
        raise SystemExit("Add dirt works on redraw versions")
    sub = "retry/" if m.group(1) == "retry" else ""
    for name in (f"klein_s{m.group(2)}_rgba.png", f"klein_s{m.group(2)}_cm.png", f"klein_s{m.group(2)}.png"):
        if (d / sub / name).exists():
            return d / sub / name
    raise SystemExit("version not found")


def _open(path):
    from PIL import Image
    im = Image.open(path)
    return np.asarray(im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") and "A" in im.getbands() else "RGB"))


def _preview(arr, size=320):
    from PIL import Image
    h, w = arr.shape[:2]
    sc = size / max(h, w); tw, th = max(1, round(w * sc)), max(1, round(h * sc))
    im = Image.fromarray(arr).resize((tw, th), Image.LANCZOS)
    if arr.shape[2] == 4:
        bg = Image.new("RGBA", (tw, th), (205, 210, 220, 255)); im = Image.alpha_composite(bg, im.convert("RGBA")).convert("RGB")
    return im.convert("RGB")


def options(uid, vid):
    src = _src(uid, vid)
    out = UN / uid / "dirt" / re.sub(r"[^A-Za-z0-9]+", "_", vid); out.mkdir(parents=True, exist_ok=True)
    arr = _open(src); res = []
    big = max(arr.shape[:2]) > 1400
    for pid, label, kw in PRESETS:
        f = out / f"opt_{pid}.jpg"
        if not f.exists() or f.stat().st_mtime < src.stat().st_mtime:
            work = arr if not big else cv2.resize(arr, (arr.shape[1] // 2, arr.shape[0] // 2), interpolation=cv2.INTER_AREA)
            _preview(render(work, kw, uid)).save(f, quality=88)
        res.append({"id": pid, "label": label, "t": int(f.stat().st_mtime)})
    return res


def apply(uid, pid, vid):
    from PIL import Image
    if pid not in LABELS:
        raise SystemExit("unknown option")
    src = _src(uid, vid); d = UN / uid; cdir = d / "custom"; cdir.mkdir(exist_ok=True)
    arr = _open(src)
    res = render(arr, next(p[2] for p in PRESETS if p[0] == pid), uid)
    jp = cdir / "custom.json"
    try:
        items = json.loads(jp.read_text()).get("items", [])
    except (OSError, ValueError):
        items = []
    old = next((it for it in items if it.get("dirt_of") == vid), None)                  # one dirt version per source: replace the earlier one
    if old:
        n = int(old["id"].split(":")[1]); items = [it for it in items if it is not old]
    else:
        n = max([int(it["id"].split(":")[1]) for it in items] + [0]) + 1
    if len(items) > 20:
        raise SystemExit("too many versions on this unit")
    fn = f"up{n}.png"; tmp = cdir / (fn + ".tmp.png")
    Image.fromarray(res).save(tmp); os.replace(tmp, cdir / fn)
    items.append({"id": f"custom:{n}", "file": fn, "rgba": res.shape[2] == 4, "alpha": "same as the clean version", "size": [res.shape[1], res.shape[0]], "ts": time.time(),
                  "replaces": vid, "dirt_of": vid, "label": "Dirt: " + LABELS[pid]})
    tmpj = jp.with_suffix(".tmp"); tmpj.write_text(json.dumps({"items": items})); os.replace(tmpj, jp)
    return {"applied": pid, "id": f"custom:{n}"}


if __name__ == "__main__":
    try:
        if sys.argv[1] == "options":
            print(json.dumps(options(sys.argv[2], sys.argv[3])))
        elif sys.argv[1] == "apply":
            print(json.dumps(apply(sys.argv[2], sys.argv[3], sys.argv[4])))
    except SystemExit as e:
        print(json.dumps({"error": str(e)})); sys.exit(0)
