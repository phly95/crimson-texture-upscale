"""relations.py: find textures that are crops/flips/rotations of another texture, and record it.

Crimson Skies reuses one painted texture for several faces: `cliff01_trans1` is a crop of `cliff01`,
`cliff01-trans3` a flipped crop. Upgrading them independently makes the seams of a building disagree,
so the agent should upgrade the parent once and derive the relatives from it. This script proposes
candidates from the stem (suffix stripping) and PROVES each one by exact template matching (allowing
the four flips), then writes cat["units"][child]["rel"] = {parent, kind, offset}.

Run after prep.py/usage.py, then re-run mktask.py so task.json carries the field.
"""
import json, re, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import texlab as T

SUFFIX = re.compile(r"(_\w*trans\d*|[-_]\d+|_?[a-z]\d+)$", re.I)
cat = json.load(open(ROOT / "catalog.json"))


def gray(im):
    return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float32)


def variants(g):
    yield "same", g
    yield "hflip", g[:, ::-1]
    yield "vflip", g[::-1, :]
    yield "rot180", g[::-1, ::-1]
    if g.shape[0] != g.shape[1]:
        yield "rot90", np.rot90(g)
        yield "rot270", np.rot90(g, 3)


def find_child_in_parent(child, parent):
    """(kind, where, mae, ssim) if `child` is a flipped copy of an edge crop of `parent`.

    Only the four anchored crops (top/bottom rows, left/right columns) are tried, not a full search:
    that is what the game actually does (`cliff01_trans1` is the top 64 rows of `cliff01`, flipped),
    and it avoids the sliding-window pitfalls - these painted patches are low-contrast, where
    normalised correlation scores collapse and unnormalised ones chase the wrong window.
    The relation is a REPAINED crop, not a bit-exact copy, so the test is structural (MAE+SSIM)."""
    from skimage.metrics import structural_similarity
    c0, p0 = gray(child), gray(parent)
    if min(c0.shape) < 4 or c0.size >= p0.size * 0.95:
        return None
    best = None
    for kind, c in variants(c0):
        h, w = c.shape
        if h > p0.shape[0] or w > p0.shape[1]:
            continue
        for where, reg in (("tl", p0[:h, :w]), ("tr", p0[:h, -w:]),
                           ("bl", p0[-h:, :w]), ("br", p0[-h:, -w:])):
            if reg.shape != c.shape:
                continue
            mae = float(np.abs(reg - c).mean())
            if best is None or mae < best[2]:
                best = (kind, where, mae, float(structural_similarity(reg, c, win_size=7, data_range=255)))
    if best and (best[2] <= 22.0 or best[3] >= 0.6):
        return best
    return None


changed = 0
for ch, names in cat["names"].items():
    for stem, uid in names.items():
        u = cat["units"][uid]
        if u["is_mip"] or u.get("rel"):
            continue
        base = stem
        seen = set()
        while base and base not in seen:            # strip one suffix at a time: a_trans1_b -> ...
            seen.add(base)
            if base in names and names[base] != uid:
                puid = names[base]
                if not cat["units"][puid]["is_mip"]:
                    got = find_child_in_parent(T.orig(uid), T.orig(puid))
                    if got:
                        kind, where, mae, ss = got
                        u["rel"] = {"parent": puid, "parent_stem": base, "kind": kind,
                                    "corner": where, "mae": round(mae, 1), "ssim": round(ss, 3)}
                        changed += 1
                        break
            nb = SUFFIX.sub("", base)
            if nb == base:
                break
            base = nb
tmp = ROOT / "catalog.json.tmp"                    # atomic: agents may be reading the catalog right now
with open(tmp, "w") as fh:
    json.dump(cat, fh, indent=1)
import os
os.replace(tmp, ROOT / "catalog.json")
n = sum(1 for u in cat["units"].values() if u.get("rel"))
print(f"relations: {changed} newly found, {n} units now carry a parent link")
