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
UI pack: ui_textures/<original folder>/<name>.png as before (chosen version, else the game's original).
"""
import json, os, shutil, sys
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

SOFT_LO, SOFT_HI = 8 / 255, 247 / 255


def bleed(rgb, a):
    """rgb of pixels with alpha 0 := nearest pixel with alpha > 0 (so no key colour survives under transparency)"""
    hole = a == 0
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
    rgb = bleed(rgb, a)
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

    # --- UI (unchanged logic)
    X = Path(os.path.expanduser("~/Games/CSVM/extracted")); n_ui = 0
    for uid, u in ui_cat["units"].items():
        r = decs.get(uid, {}); rel = Path(u["ui_src"]); d = out / "ui_textures" / rel.parent; d.mkdir(parents=True, exist_ok=True)
        src = ew.get_chosen_file(uid, r["chosen"]) if r.get("decision") == "approve" and r.get("chosen") else None
        if src and src.exists():
            shutil.copy2(src, d / f"{rel.stem}.png")
        else:
            shutil.copy2(X / rel, d / rel.name)
        n_ui += 1
    print(f"ui: {n_ui} files")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else ROOT / "release_v200")
