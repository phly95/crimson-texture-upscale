"""build_pack.py: assemble the HD texture pack from accepted agent outputs, and install it.

  build_pack.py build               assemble pack/<CH>/<tierN>/ (complete folders: every original file,
                                          upgraded textures substituted, mip siblings of upgraded ones dropped)
  build_pack.py status                    what is upgraded / kept / waiting for ESRGAN, per chapter
  build_pack.py install                   symlink pack/<CH>/<tierN> next to the game's zip (the engine prefers an
                                          unpacked folder to the zip; nothing is overwritten)
  build_pack.py uninstall                 remove those symlinks (only symlinks that point into this pack)

A unit ships when its latest decision in pipeline/reviews.jsonl is 'approve'; the chosen version (klein/retry/magenta/photo/custom) is
conformed to the target size and the original's colour type. Everything else keeps its original file, so a partial pack is always valid.
"""
import json, os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAME = Path(os.path.expanduser("~/Games/CSVM/extracted"))
PACK = ROOT / "pack"


PIPE = ROOT / "pipeline"
CACHE = ROOT / "pack_cache"            # conformed final PNGs (target size, original colour type); survives rebuilds of pack/


def decisions():
    """uid -> latest review record (reviews.jsonl is append-only; the last line for a unit wins)"""
    last = {}
    for ln in (PIPE / "reviews.jsonl").read_text().splitlines():
        try:
            r = json.loads(ln); last[r["uid"]] = r
        except (ValueError, KeyError):
            continue
    return last


def version_file(uid, vid):
    """the picture the review page showed for version id vid ('klein:1', 'retry:2', 'magenta:1', 'photo:1', 'custom:2')"""
    d = PIPE / "units" / uid
    kind, n = vid.split(":")
    if kind == "custom":
        for it in json.load(open(d / "custom" / "custom.json")).get("items", []):
            if it["id"] == vid:
                return d / "custom" / it["file"]
        return None
    sub = {"klein": "", "retry": "retry", "magenta": "magenta", "photo": "photo"}[kind]
    for suf in (("_rgba", "_cm", "") if kind != "magenta" else ("_rgba",)):
        f = d / sub / f"klein_s{n}{suf}.png"
        if f.exists():
            return f
    return None


def alpha_soft(a):
    import numpy as np
    nz = a > 0
    return bool(nz.sum()) and (a[nz] >= 200).mean() < 0.45          # the engine's AlphaIsSoft test


def orig_path(u):
    stem, chs = next(iter(u["stems"].items()))
    return ROOT / "originals" / chs[0] / f"{stem}.png"


def conform(uid, src, cat):
    """final file for a unit: target size, RGBA when the original had alpha else RGB; returns (path, class_flipped)"""
    import numpy as np
    from PIL import Image
    u = cat["units"][uid]
    W, H = u["target"]
    dst = CACHE / f"{uid}.png"
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst, False
    im = Image.open(src).convert("RGBA")
    if im.size != (W, H):
        im = im.resize((W, H), Image.LANCZOS)
    flip = False
    if u["has_alpha"]:
        o = np.asarray(Image.open(orig_path(u)).convert("RGBA").getchannel("A"))
        flip = alpha_soft(o) != alpha_soft(np.asarray(im.getchannel("A")))
    else:
        im = im.convert("RGB")
    CACHE.mkdir(exist_ok=True)
    im.save(dst, optimize=False)
    return dst, flip


def accepted(cat=None, verbose=False):
    """uid -> path of the approved, conformed PNG.  Only 'approve' ships; 'original', reject, undo and unreviewed units keep the game's original file."""
    cat = cat or json.load(open(ROOT / "catalog.json"))
    best, flips, missing = {}, [], []
    for uid, r in decisions().items():
        if r.get("decision") != "approve" or uid not in cat["units"] or not r.get("chosen"):
            continue
        src = version_file(uid, r["chosen"]) if (PIPE / "units" / uid).is_dir() else None
        if src is None:
            missing.append(uid); continue
        p, fl = conform(uid, src, cat)
        best[uid] = p
        if fl:
            flips.append(uid)
    if verbose or flips or missing:
        print(f"approved with a file: {len(best)}; no file found: {len(missing)} {missing[:5]}; alpha class changed (soft<->hard): {len(flips)} {flips[:8]}")
    return best


def pending_esrgan():
    return set()


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)                    # shared across chapters, no extra disk
    except OSError:
        shutil.copy2(src, dst)


def build(pack=PACK):
    cat = json.load(open(ROOT / "catalog.json"))
    acc = accepted(cat)
    shutil.rmtree(pack, ignore_errors=True)
    total = {"chapters": 0, "files": 0, "upgraded": 0}
    for ch, names in cat["names"].items():
        tier = cat["tiers"][ch]
        out = pack / ch / tier
        out.mkdir(parents=True)
        up = {stem for stem, uid in names.items() if uid in acc}
        drop = set()
        for stem in up:                      # engine rule: X_1/X_2 are mips of X; sizes must match exactly
            for lvl in (1, 2):
                if f"{stem}_{lvl}" in names:
                    drop.add(f"{stem}_{lvl}")
        for f in (ROOT / "originals" / ch).iterdir():
            if f.suffix.lower() == ".png" and f.stem in drop:
                continue
            if f.suffix.lower() == ".png" and f.stem in up:
                link_or_copy(acc[names[f.stem]], out / f.name)
                total["upgraded"] += 1
            else:
                link_or_copy(f, out / f.name)
            total["files"] += 1
        total["chapters"] += 1
    size = sum(p.stat().st_size for p in pack.rglob("*.png") if p.is_file())
    print(f"built {pack}: {total['chapters']} chapters, {total['files']} files, "
          f"{total['upgraded']} upgraded entries, {len(acc)} unique upgraded textures "
          f"(hard-linked duplicates not de-duplicated in this size figure: {size / 1e6:.0f} MB)")


def status():
    cat = json.load(open(ROOT / "catalog.json"))
    acc, esr = accepted(cat), pending_esrgan()
    work = [u for u, v in cat["units"].items() if not v["is_mip"] and not v["is_placeholder"]]
    print(f"work units {len(work)}: upgraded {sum(u in acc for u in work)}, "
          f"waiting for ESRGAN {sum(u in esr and u not in acc for u in work)}, "
          f"untouched {sum(u not in acc and u not in esr for u in work)}")
    for ch, names in cat["names"].items():
        us = {uid for uid in names.values() if uid in work}
        print(f"  {ch:4s} {cat['tiers'][ch]:11s} upgraded {sum(u in acc for u in us):4d} / {len(us)}")


def install():
    cat = json.load(open(ROOT / "catalog.json"))
    for ch, tier in cat["tiers"].items():
        src = PACK / ch / tier
        dst = GAME / ch / tier
        if not src.is_dir():
            print(f"skip {ch}: nothing built"); continue
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            print(f"ok   {ch}: already installed"); continue
        if dst.exists() or dst.is_symlink():
            print(f"SKIP {ch}: {dst} already exists and is not ours"); continue
        os.symlink(src, dst)
        print(f"link {dst} -> {src}")


def uninstall():
    for ch_dir in sorted(GAME.glob("C*")):
        for p in ch_dir.glob("rtexture*"):
            if p.is_symlink() and str(os.readlink(p)).startswith(str(PACK)):
                p.unlink()
                print(f"removed {p}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"build": build, "status": status, "install": install, "uninstall": uninstall}[cmd]()
