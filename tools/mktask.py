"""mktask.py <task_id> [<task_id> ...]  - build tasks/<id>/ (orig/, sheet.png, task.json) for the agents."""
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image, ImageDraw
import texlab as T

ROOT = T.ROOT
tasks = {t["id"]: t for t in json.load(open(ROOT / "tasks" / "tasks.json"))}


def flat_score(img):
    """Distinct colours after 4-bit quantisation, over visible pixels: flat graphics are low."""
    a = np.asarray(img.convert("RGBA"))
    vis = a[..., 3] > 32
    if vis.sum() == 0:
        return 0
    q = (a[..., :3][vis] >> 4).astype(np.uint16)
    return int(len(np.unique(q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2])))


def build(tid):
    t = tasks[tid]
    d = ROOT / "tasks" / tid
    if d.exists():
        shutil.rmtree(d)
    (d / "orig").mkdir(parents=True)
    units = []
    for uid in t["units"]:
        i = T.info(uid)
        o = T.orig(uid)
        shutil.copy(T.orig_path(uid), d / "orig" / f"{uid}.png")
        rgb = o.convert("RGB")
        mean = np.asarray(rgb).reshape(-1, 3).mean(0)
        n = flat_score(o)
        units.append({
            "uid": uid, "stem": i["stem"], "size": [i["w"], i["h"]], "target": i["target"],
            "has_alpha": i["has_alpha"], "alpha_class": i.get("alpha"),
            "alpha_is_soft": T.soft_alpha(o) if i["has_alpha"] else None,
            "tileable_guess": bool(T.seam_ratio(rgb) <= 1.6), "seam_ratio": round(T.seam_ratio(rgb), 2),
            "mean_rgb": [round(float(x), 1) for x in mean],
            "distinct_colours_4bit": n, "look": "flat-graphic-likely" if n <= 40 else "photo/painted-likely",
            "also_named": sorted(s for s in i["stems"] if s != i["stem"]),
            "chapters": i["chapters"], "flipbook_id": i.get("flip"),
            "usage": i.get("usage"), "rel": i.get("rel"),
        })
    json.dump({"task": tid, "kind": t["kind"], "family": t["family"], "units": units},
              open(d / "task.json", "w"), indent=1)
    # contact sheet: originals enlarged with nearest-neighbour so their real pixels are visible
    cell, cols = 192, 4
    rows = (len(units) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell + 8), rows * (cell + 30)), (30, 30, 36))
    dr = ImageDraw.Draw(sheet)
    for k, u in enumerate(units):
        o = T.orig(u["uid"])
        s = max(1, min(cell // o.width, cell // o.height))
        o = o.resize((o.width * s, o.height * s), Image.NEAREST)
        bg = Image.new("RGBA", o.size, (255, 0, 255, 255))
        bg.alpha_composite(o)
        x, y = (k % cols) * (cell + 8), (k // cols) * (cell + 30)
        sheet.paste(bg.convert("RGB"), (x, y))
        dr.text((x, y + cell + 2), f"{u['stem'][:28]}", fill=(240, 240, 240))
        dr.text((x, y + cell + 14), f"{u['size'][0]}x{u['size'][1]} -> {u['target'][0]}x{u['target'][1]}"
                + (" alpha" if u["has_alpha"] else ""), fill=(170, 200, 255))
    sheet.save(d / "sheet.png")
    return d


if __name__ == "__main__":
    for tid in sys.argv[1:]:
        print(build(tid))
