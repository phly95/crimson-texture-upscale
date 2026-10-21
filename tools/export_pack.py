"""export_pack.py: build the v2.0.0 release folders from the review decisions (replaces export_winners.py for the world pack).

  venv/bin/python tools/export_pack.py [OUT]       default OUT = release_v200/   (world_textures/, ui_textures/)

World pack (CSVM HdTextures lookup: a bare name applies to every chapter, <CH>/<tier>/<name> beats it in that chapter only):
  * a name with one picture in every chapter -> world_textures/<name>.png
  * a name with several pictures (45 names: cliff01, cloud1, wtr000NN ...) -> the largest chapter group's picture is the bare file,
    every other group gets world_textures/<CH>/<tier>/<name>.png for each of its chapters (originals included when that group kept the
    original, otherwise the bare file would override it).
Cutouts get two clean-ups on the way out (flux_fix tolerance is checked afterwards and printed):
  * colour under alpha 0 is bled from the nearest opaque pixel (the redraws kept the green key colour there, which fringes when filtered)
  * soft edge band restore: where the original had a soft alpha band (glow disk, feathered rim) that the redraw's crisp key removed,
    alpha = max(redraw, original soft alpha resized) within a short distance of the redraw's silhouette, with the original's colour there
UI pack: ui_textures/<original folder>/<name>.png as before (chosen version, else the game's original). The memento polaroids (rimage/ms_p_*, momento_temp) are
fitted to the original's outline and momento_shad (the sheet stack under them) is the original smoothly upscaled, see outline_fit.py. SB_KILLMARKERCOMBINED
(the kill-tally stamps) is tile_job/killmarkers/atlas.png from tile_stamps.py.
"""
import json, os, re, shutil, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "flux_fix"))
import build_pack as bp
import export_winners as ew
import tol
import outline_fit as of

SOFT_LO, SOFT_HI = 8 / 255, 247 / 255
FAINT = 32                                                      # hard-edged cutouts: alpha below this is key-colour haze, not part of the picture
FOLIAGE = re.compile(r"tree|spruce(?!goose)|palm|brush|bush|terpat|shipwreck|cliff0\d_trans|grass|leaf|vine|ivy|moss|hedge|shrub", re.I)   # pictures that can hold green-screen specks
CONIFER = re.compile(r"tree|spruce(?!goose)", re.I)                 # needle trees: a lighter-than-inside edge is a halo (palms, thistles and grass have genuinely pale leaves)
RIM = 14                                                        # foliage edge band (px at 1024) in which a lighter-than-inside rim is pulled towards the foliage colour
LEAK_MAX = 150                                                  # largest saturated-green speck (px at 1024) still treated as missed key colour
JPG_DEFAULT = re.compile(r"^(ms_p_|sb_03_01_base$|sb_03_01_jack$|sb_04_01_lifesaver$)", re.I)    # PNG/JPG scrapbook pairs where the JPG is a photograph with a plain white border: it serves the page's small copy too
OFF = 40                                                        # mean RGB distance from the nearest solid pixel's colour that marks an edge pixel as key-colour leak
SOLID = 224                                                     # ... and colour under anything less opaque than this is taken from the nearest solid pixel


def bleed(rgb, a, solid=1):
    """rgb of pixels with alpha < solid (default: alpha 0) := nearest pixel with alpha >= solid (so no key colour survives under transparency)"""
    hole = a < solid
    if not hole.any() or hole.all():
        return rgb
    _, lab = cv2.distanceTransformWithLabels(hole.astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    # every zero pixel of the input (= every opaque pixel) is its own label; holes take the label of their nearest one
    lut = np.zeros(lab.max() + 1, np.int64)
    ys, xs = np.nonzero(~hole)
    flat = lab[~hole]
    lut[flat] = np.arange(len(flat))
    idx = lut[lab[hole]]
    out = rgb.copy()
    out[hole] = rgb[ys[idx], xs[idx]]
    return out


def wide_soft(o):
    """pixels of a genuinely soft region of the original alpha (glow disk, feathered rim): soft values that are not just the 1-2 px antialiasing
    ring next to a solid area. o = alpha 0..1 at the original's size"""
    soft = (o > SOFT_LO) & (o < SOFT_HI)
    solid = cv2.dilate((o >= SOFT_HI).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    return soft & ~solid


def soft_loss(orig_a, hd_a):
    """(share of the original's wide-soft alpha mass missing from the redraw, wide-soft mask) compared at the original's size"""
    o = orig_a.astype(np.float32) / 255
    w = wide_soft(o)
    if w.sum() < max(8, 0.004 * o.size):
        return 0.0, w
    h = np.asarray(Image.fromarray(hd_a).resize(orig_a.shape[::-1], Image.BOX), np.float32) / 255
    return float(np.maximum(0, o[w] - h[w]).sum() / o[w].sum()), w


def clean_cutout(hd_path, orig_path):
    """returns (RGBA image, notes)"""
    hd = np.asarray(Image.open(hd_path).convert("RGBA"))
    H, W = hd.shape[:2]
    og = Image.open(orig_path).convert("RGBA")
    notes = []
    loss, wmask = soft_loss(np.asarray(og.getchannel("A")), hd[..., 3])
    rgb, a = hd[..., :3].copy(), hd[..., 3].copy()
    band = np.zeros((H, W), bool)                              # pixels whose soft alpha / colour were restored from the original (left alone below)
    if loss > 0.4:
        up = og.resize((W, H), Image.BICUBIC)
        oa = np.asarray(up.getchannel("A"), np.float32) / 255
        orgb = np.asarray(up.convert("RGB"))
        wm = np.asarray(Image.fromarray(wmask.astype(np.uint8) * 255).resize((W, H), Image.BICUBIC)) > 127
        band = wm & (oa > SOFT_LO)
        newa = np.where(band, np.maximum(a / 255, oa), a / 255)
        w = np.where(newa > 0, np.clip((newa - a / 255) / np.maximum(newa, 1e-3), 0, 1), 0)[..., None]
        base = bleed(rgb, a)
        rgb = (base * (1 - w) + orgb * w).astype(np.uint8)
        a = (newa * 255 + .5).astype(np.uint8)
        notes.append(f"soft band restored (loss {loss:.2f})")
    # Key green missed by the keyer: small saturated-green specks and clusters between leaves and needles (firtree1/2, palm1-3, bush1/2, brush3/4 ...). The
    # original has next to none, so a speck (a green-dominant component of at most LEAK_MAX px) in a redraw of it is leftover background colour. It and the
    # greenish pixels around it are painted over from the neutral foliage around them. Never made transparent: bigger lime shapes are sunlit leaves the
    # redraw drew on purpose. Only foliage pictures (by name), only where the original has green of its own, and anything larger than a speck, are left alone.
    def greeny(img, thr):
        c = img.astype(np.int16)
        return (c[..., 3] >= 128) & (c[..., 1] - np.maximum(c[..., 0], c[..., 2]) > thr)
    seeds = greeny(np.dstack([rgb, a]), 40).astype(np.uint8)
    og_arr = np.asarray(og)
    foliage = FOLIAGE.search(Path(orig_path).stem) is not None
    if foliage and seeds.any() and greeny(og_arr, 40).mean() < 2e-4 and seeds.mean() < 0.02:
        k_, lab, stt, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
        c16 = rgb.astype(np.int16)
        sat = c16[..., 1] - np.maximum(c16[..., 0], c16[..., 2])                      # how green-dominant each pixel is
        calm = (a >= 128) & (seeds == 0) & (sat < 25)                                  # ordinary needle / leaf pixels
        core = np.zeros((H, W), bool)
        for i in range(1, k_):
            x0, y0, bw, bh, area = stt[i]
            if area > LEAK_MAX:
                continue
            ys, xs = slice(max(0, y0 - 6), y0 + bh + 6), slice(max(0, x0 - 6), x0 + bw + 6)
            m = lab[ys, xs] == i
            ring = (cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=4) > 0) & ~m & (a[ys, xs] >= 128) & (seeds[ys, xs] == 0)
            # a speck is vividly green (neon) and sits in calm foliage; a green leaf (or part of one) is duller and has green around it
            if ring.sum() >= 6 and np.percentile(sat[ys, xs][m], 90) >= 50 and sat[ys, xs][ring].mean() < 22:
                core[ys, xs] |= m
        if core.any():
            near = cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3) > 0
            drop = core | (near & greeny(np.dstack([rgb, a]), 18))
            # paint over from calm opaque neighbours only (normalised blur; transparent pixels carry key colour and must not be sampled)
            valid = ((a >= 200) & ~drop & (sat < 14)).astype(np.float32)
            filled = rgb.astype(np.float32); done = np.zeros((H, W), bool)
            for sg in (2, 4, 8):
                den = cv2.GaussianBlur(valid, (0, 0), sg)
                num = np.stack([cv2.GaussianBlur(rgb[..., k].astype(np.float32) * valid, (0, 0), sg) for k in range(3)], -1)
                ok = drop & ~done & (den > 0.03)
                filled[ok] = num[ok] / den[ok][:, None]; done |= ok
            rgb = np.where(done[..., None], np.clip(filled + .5, 0, 255).astype(np.uint8), rgb)
            notes.append(f"key green specks painted over ({int(core.sum())} px)")
    if (a >= SOLID).any():
        # The redraw's keyed edge can keep a haze / fringe in the key colour at low alpha (near-white or magenta); mipmaps and filtering pull it into the
        # leaves as a pale outline, and bleeding from alpha > 0 copied it under the transparent pixels too. Only pixels whose colour is clearly off
        # against the nearest solid pixel are touched (honest shadows, glass, antialiasing and the restored soft bands stay).
        soft_orig = wide_soft(np.asarray(og.getchannel("A"), np.float32) / 255).sum() >= max(8, 0.004 * og.width * og.height)
        hard = not soft_orig or foliage                              # foliage pictures: haze and edge clean-up even when the original has a soft shadow band
        haze = np.zeros((H, W), bool)
        kx, ky = -(-W // og.width), -(-H // og.height)
        if hard:                                                    # hard-edged original (trees, wire, signs): faint off-colour alpha is haze, and its empty edge texels stay empty
            haze = (a > 0) & (a < FAINT) & ~band & (np.abs(rgb.astype(np.int16) - bleed(rgb, a, SOLID).astype(np.int16)).mean(-1) > OFF)
            a[haze] = 0
            # faint patches far from any real foliage (dark smudges the keyer left in the empty area) are stray: anything under 64/255 opacity that is not within
            # a few px of foliage goes transparent. The restored soft shadow band is excluded.
            near_solid = cv2.dilate((a >= 64).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(6, 2 * kx)) > 0
            stray = (a > 0) & (a < 64) & ~near_solid & ~band
            a[stray] = 0
            oa = np.asarray(og.getchannel("A"))
            for side, line, strip in (("l", oa[:, 0], (slice(None), slice(0, kx))), ("r", oa[:, -1], (slice(None), slice(W - kx, W))),
                                      ("t", oa[0, :], (slice(0, ky), slice(None))), ("b", oa[-1, :], (slice(H - ky, H), slice(None)))):
                faint = a[strip] < 128
                if line.max() <= 4 and (a[strip][faint] > 0).any():  # the thin line seen on a tiled/clamped sprite
                    a[strip] = np.where(faint, 0, a[strip]); notes.append(f"cleared {side} edge strip")
        ref = bleed(rgb, a, SOLID)
        rgb = bleed(rgb, a)
        near_hole = cv2.dilate((a == 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(2, kx)) > 0
        r16, g16, b16 = (rgb[..., k].astype(np.int16) for k in range(3))
        keylike = (np.minimum(r16, b16) - g16 > 50) | ((g16 > 180) & (r16 < 90) & (b16 < 90))      # magenta / pure green key colour
        off = np.abs(rgb.astype(np.int16) - ref.astype(np.int16)).mean(-1) > OFF
        # translucent areas (glass, splashes, wire shadows) are never repainted; under fully transparent pixels (invisible, only filtering sees them) a
        # hard-edged cutout takes the solid neighbour's colour wherever it is off, a soft one only where it is key-coloured
        if foliage:                                                  # foliage has no translucent glass: every non-solid pixel at the edge takes its solid neighbour's colour (classic edge bleed), so no pale rim shows on dark backdrops
            near_fringe = cv2.dilate((a == 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(6, 3 * kx)) > 0
            fix = (a < SOLID) & near_fringe & ~band | (a == 0) & (off if hard else keylike)
        else:
            fix = (a < SOLID) & near_hole & ~band & keylike | (a == 0) & (off if hard else keylike)
        rgb = np.where(fix[..., None], ref, rgb)
        if CONIFER.search(Path(orig_path).stem):
            # The magenta redraws of the conifers come out with a pale, feathered rim (grey-lilac, up to ~20 px wide at 1024) around the needles and leaves: klein paints a light glow
            # at the screen edge and the key keeps it. It shows as a white outline on a dark backdrop, which the originals do not have. In the edge band (RIM px at
            # 1024) every pixel that is lighter than the foliage inside is pulled towards the colour of that foliage; darker detail is left alone.
            band_w = max(6, round(1.5 * RIM * max(W, H) / 1024))
            dist = cv2.distanceTransform((a >= 32).astype(np.uint8), cv2.DIST_L2, 3)       # px to the nearest empty pixel
            inner = (a >= SOLID) & (dist > band_w)
            if inner.any():
                deep = bleed(rgb, np.where(inner, 255, 0).astype(np.uint8), 1).astype(np.float32)
                ring = (a > 0) & (dist <= band_w)                              # includes the restored shadow band: dark there, so a hue pull does not show, but pale needle tips in it do
                cur = rgb.astype(np.float32)
                # the key also leaves the rim greyer / bluer than the foliage (the magenta is un-mixed to neutral, and klein draws the outer needle sprays pale blue-grey):
                # the rim takes the tree's overall hue (median colour of the deep foliage) at its own brightness
                core = (a >= SOLID) & (dist > 1.5 * band_w)
                gref = np.median(cur[core], axis=0) if core.sum() > 500 else deep[inner].mean(0)
                tint = (gref[1] - gref[2]) - (cur[..., 1] - cur[..., 2])                       # > 0: bluer / greyer than the tree overall
                ph = np.where(ring, np.clip((tint - 6) / 20, 0, 1), 0)[..., None]
                hue = gref[None, None, :] * np.clip(cur.mean(-1) / max(float(gref.mean()), 1.0), 0.2, 3.0)[..., None]
                cur = cur * (1 - ph) + np.clip(hue, 0, 255) * ph
                ex = cur.mean(-1) - deep.mean(-1)
                pull = np.where(ring, np.clip((ex - 8) / 30, 0, 1), 0)[..., None]
                rgb = np.clip(cur * (1 - pull) + deep * pull + .5, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, a]), "RGBA"), notes


def run(out_dir):
    cat = json.load(open(ROOT / "catalog.json"))
    ui_cat = json.load(open(ROOT / "ui_catalog.json"))
    decs = ew.decisions()
    acc = {}                                                    # uid -> the approved picture at the size it was made (not resized: v1.0.0 shipped them as made)
    for uid, r in decs.items():
        if r.get("decision") == "approve" and r.get("chosen") and uid in cat["units"] and (bp.PIPE / "units" / uid).is_dir():
            f = bp.version_file(uid, r["chosen"])
            if f is not None:
                acc[uid] = f
    out = Path(out_dir)
    shutil.rmtree(out, ignore_errors=True)
    (out / "world_textures").mkdir(parents=True); (out / "ui_textures").mkdir()
    cache = ROOT / "pack_cache" / "clean"; cache.mkdir(parents=True, exist_ok=True)

    def final_file(uid):
        """the picture to ship for a unit: cleaned approved redraw, else the game's original"""
        u = cat["units"][uid]
        if uid in acc:
            if not u["has_alpha"]:
                return acc[uid], []
            c = cache / f"{uid}.png"
            if c.exists() and c.stat().st_mtime >= acc[uid].stat().st_mtime:
                return c, ["cached"]
            im, notes = clean_cutout(acc[uid], bp.orig_path(u))
            im.save(c); return c, notes
        return bp.orig_path(u), []

    # --- world: bare vs chapter entries
    by_stem = defaultdict(lambda: defaultdict(list))            # stem -> uid -> [chapters]
    for ch, names in cat["names"].items():
        for stem, uid in names.items():
            u = cat["units"][uid]
            if not (u["is_mip"] or u["is_placeholder"]):
                by_stem[stem][uid].append(ch)
    n_bare = n_ch = 0; report = []
    for stem, groups in sorted(by_stem.items()):
        ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        for k, (uid, chs) in enumerate(ordered):
            src, notes = final_file(uid)
            ext = ".png"
            if k == 0:
                shutil.copy2(src, out / "world_textures" / f"{stem}{ext}"); n_bare += 1
            else:
                for ch in chs:
                    d = out / "world_textures" / ch / cat["tiers"][ch]; d.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, d / f"{stem}{ext}"); n_ch += 1
            if notes and notes != ["cached"]:
                report.append((stem, notes))
    print(f"world: {n_bare} bare files, {n_ch} chapter entries; {len(by_stem)} names, {sum(len(g) > 1 for g in by_stem.values())} with several pictures")
    for s, n in report:
        print("   cleaned", s, n)

    # --- tolerance gate over what ships
    bad = []
    for uid in acc:
        u = cat["units"][uid]
        if u["is_mip"] or u["is_placeholder"] or not u["has_alpha"]:
            continue
        m = tol.metrics(bp.orig_path(u), final_file(uid)[0])
        if not m["pass"]:
            bad.append((u["stem"], m["diff"], m["dcov"]))
    print(f"tolerance gate: {len(bad)} shipped cutouts outside diff<=0.12 / |dcov|<=0.15")
    for b in sorted(bad, key=lambda x: -x[1]):
        print("   ", *b)

    # --- UI. Scrapbook items often exist as a .PNG (the small cut-out on the page), a .JPG (the large picture the zoom shows) and a .TIF; the engine matches an HD file by
    # name without extension and draws it at the matching original's size, so one file per stem is stretched over the other variant (SB_03_01_BASE's zoom drew the redraw of
    # its rotated thumbnail at 4:3). Where both a PNG and a JPG original exist both are shipped under their own extension, for an engine that keys by extension; for the
    # engine as it is, the later zip entry wins, and ui_default_last.json (read by zip_pack.py) says which one that is: the JPG for photographs with a plain white border and wherever the two originals' aspect ratios differ by more than 25 percent (the zoom is the large picture, and the one a wrong aspect ruins),
    # the PNG for the other cut-outs, whose JPG carries a desk backdrop. The TIF is not used by any screen.
    X = Path(os.path.expanduser("~/Games/CSVM/extracted")); n_ui = 0
    groups = defaultdict(dict)                                  # (folder, stem lower) -> {ext: uid}
    for uid, u in ui_cat["units"].items():
        rel = Path(u["ui_src"]); groups[(str(rel.parent), rel.stem.lower())][rel.suffix.lower()] = uid
    default_last = {}
    for uid, u in ui_cat["units"].items():
        r = decs.get(uid, {}); rel = Path(u["ui_src"]); d = out / "ui_textures" / rel.parent; d.mkdir(parents=True, exist_ok=True)
        src = ew.get_chosen_file(uid, r["chosen"]) if r.get("decision") == "approve" and r.get("chosen") else None
        src = src if src and src.exists() else None
        ext = rel.suffix.lower(); stem = rel.stem.lower()
        variants = groups[(str(rel.parent), stem)]
        both = ".png" in variants and ".jpg" in variants
        if both and ext not in (".png", ".jpg"):
            continue
        if len(variants) > 1 and not both and ext not in (".png", ".jpg") and (".png" in variants or ".jpg" in variants):
            continue
        if rel.parent.name == "rimage" and (stem.startswith("ms_p_") or stem == "momento_temp") and src:
            o = np.asarray(Image.open(X / rel).convert("RGBA"))
            Image.fromarray(of.polaroid_fit(np.asarray(Image.open(src).convert("RGBA")), o)).save(d / f"{rel.stem}.png")
        elif rel.parent.name == "rimage" and stem == "momento_shad":
            o = np.asarray(Image.open(X / rel).convert("RGBA"))
            Image.fromarray(of.shadow_from_orig(o, Image.open(src).size if src else (1024, 1024))).save(d / f"{rel.stem}.png")
        elif both and ext == ".jpg":
            if src:
                im = Image.open(src); im = im.convert("RGB") if im.mode != "RGB" else im
                im.save(d / f"{rel.stem}.jpg", quality=95, subsampling=0)
            else:
                shutil.copy2(X / rel, d / f"{rel.stem}.jpg")
            pr = decs.get(variants[".png"], {})
            png_redrawn = pr.get("decision") == "approve" and pr.get("chosen")
            sizes = [Image.open(X / Path(ui_cat["units"][variants[e]]["ui_src"])).size for e in (".png", ".jpg")]
            ratio = max(sizes[0][0] / sizes[0][1] / (sizes[1][0] / sizes[1][1]), sizes[1][0] / sizes[1][1] / (sizes[0][0] / sizes[0][1]))
            default_last[f"{rel.parent}/{rel.stem}"] = "jpg" if (JPG_DEFAULT.search(rel.stem) or not png_redrawn or ratio > 1.25) and src else "png"
        elif src:
            shutil.copy2(src, d / f"{rel.stem}.png")
        else:
            shutil.copy2(X / rel, d / rel.name)
        n_ui += 1
    (out / "ui_default_last.json").write_text(json.dumps(default_last, indent=1))
    # the kill-tally stamp atlas has no unit (an atlas is not run through the pipeline): tile_stamps.py makes it, frame by frame, at 7x
    atlas = ROOT / "tile_job" / "killmarkers" / "atlas.png"
    if atlas.exists():
        d = out / "ui_textures" / "rof/ASSETS/GRAPHICS"; d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(atlas, d / "SB_KILLMARKERCOMBINED.png"); n_ui += 1
    print(f"ui: {n_ui} files")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else ROOT / "release_v200")
