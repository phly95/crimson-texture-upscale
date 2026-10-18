"""tile_job.py: tiled FLUX upscale as a per-tile, prompt-refinable job (for pictures with lettering, where one generic prompt invents or garbles text).

A job lives in tile_job/<NAME>/ :  job.json (src, scale, global hint, lock sigmas, tile/overlap, seeds), prompts.json {"s2_01": "tile prompt", ...},
  s2/ s4/  per stage: ref_YX.png (the Lanczos crop the model edits), raw_YX.png (model output), out_YX.png (colour-locked output), stage.png (assembled result).
Tile ids are  s<size>_<row><col>  (s2 = 800x600 -> 1600x1200, s4 = 1600x1200 -> 3200x2400).  Settle every s2 tile before preparing s4.

  tile_job.py new NAME SRC [--hint T] [--sigma-l N] [--sigma-ab N]   create the job, cut the s2 reference tiles
  tile_job.py gen NAME TILE [--seed N]                               (re)draw one tile with  base prompt + hint + prompts.json[TILE]
  tile_job.py all NAME STAGE                                         draw every tile of a stage that has no output yet
  tile_job.py view NAME TILE                                         write view_TILE.png: reference | redraw side by side (and 2x crops of both halves) to look at
  tile_job.py assemble NAME STAGE                                    blend the stage's tiles -> STAGE/stage.png (missing tiles fall back to the reference)
  tile_job.py next NAME                                              after s2 is assembled: cut the s4 reference tiles from s2/stage.png
  tile_job.py final NAME OUT.png                                     assemble s4 and write OUT.png
Per-tile prompts: say exactly what lettering the tile holds (verbatim, size, colour, where) or that it holds none, and forbid anything else.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tile_upscale as tu                                                        # noqa: E402

JOBS = ROOT / "tile_job"


def jd(name):
    return JOBS / name


def cfg(name):
    return json.loads((jd(name) / "job.json").read_text())


def prompts(name):
    p = jd(name) / "prompts.json"
    return json.loads(p.read_text()) if p.exists() else {}


def grid(size, T, overlap):
    W, H = size
    T = min(T, W, H)
    return T, tu.positions(W, T, overlap), tu.positions(H, T, overlap)


def stage_dir(name, stage):
    d = jd(name) / stage; d.mkdir(parents=True, exist_ok=True); return d


def cut_refs(name, stage, img):
    c = cfg(name)
    base = img.resize((img.size[0] * 2, img.size[1] * 2), Image.LANCZOS)
    T, xs, ys = grid(base.size, c["tile"], c["overlap"])
    d = stage_dir(name, stage)
    base.save(d / "base.png")
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            base.crop((x, y, x + T, y + T)).save(d / f"ref_{yi}{xi}.png")
    json.dump({"size": base.size, "T": T, "xs": xs, "ys": ys}, open(d / "grid.json", "w"))
    print(f"{stage}: {base.size[0]}x{base.size[1]}, {len(ys)} rows x {len(xs)} cols of {T}px tiles: " + ", ".join(f"{stage}_{yi}{xi}" for yi in range(len(ys)) for xi in range(len(xs))))


def new(a):
    d = jd(a.name); d.mkdir(parents=True, exist_ok=True)
    json.dump({"src": str(Path(a.src).resolve()), "hint": a.hint, "sigma_l": a.sigma_l, "sigma_ab": a.sigma_ab, "tile": 1024, "overlap": 256, "seed": 100}, open(d / "job.json", "w"), indent=1)
    if not (d / "prompts.json").exists():
        (d / "prompts.json").write_text("{}")
    img = Image.open(a.src).convert("RGB"); img.save(d / "src.png")
    cut_refs(a.name, "s2", img)


def flat_sentence(name, tile):
    """jobs with auto_flat: a tile whose reference is nearly featureless gets a positive 'plain surface' sentence (blank paper/dark surface invites invented forms, lines and clips)"""
    stage, yx = tile.split("_")
    ref = np.asarray(Image.open(jd(name) / stage / f"ref_{yx}.png").convert("L"), np.float32) / 255
    import cv2
    sm = cv2.GaussianBlur(ref, (0, 0), 5.0)
    if sm.std() > 0.05:
        return ""
    if ref.mean() > 0.45:
        return "This crop is only plain soft pale paper from edge to edge, with nothing on it: no lines, rules, boxes, clips, marks, scratches, stains, objects or lettering."
    return "This crop is only a plain dark surface from edge to edge, with nothing on it: no lines, shapes, marks, objects or lettering."


def tile_prompt(name, tile):
    c = cfg(name)
    extra = prompts(name).get(tile, "")
    if not extra and c.get("auto_flat"):
        extra = flat_sentence(name, tile)
    parts = [tu.BASE_PROMPT, c.get("hint", ""), extra]
    return " ".join(p.strip() for p in parts if p and p.strip())


def gen(name, tile, seed=None):
    stage = tile.split("_")[0]; yx = tile.split("_")[1]
    d = jd(name) / stage; c = cfg(name)
    ref = Image.open(d / f"ref_{yx}.png").convert("RGB"); T = ref.size[0]
    r = np.asarray(ref).astype(np.float32) / 255
    if r.std(axis=(0, 1)).max() < 0.012:
        out = r; print(tile, "flat: copied")
    else:
        s = seed if seed is not None else c["seed"] + sum(map(ord, tile))
        t0 = time.time()
        tu.locked_gen(d / f"ref_{yx}.png", d / f"raw_{yx}.png", tile_prompt(name, tile), T, s)
        g = np.asarray(Image.open(d / f"raw_{yx}.png").convert("RGB").resize((T, T), Image.LANCZOS)).astype(np.float32) / 255
        tu.HF_CLIP = False
        out = tu.colour_lock(g, r, c["sigma_l"], c["sigma_ab"])
        print(f"{tile}: seed {s}, {time.time() - t0:.0f}s")
    Image.fromarray((out * 255 + .5).astype(np.uint8)).save(d / f"out_{yx}.png")


def view(name, tile):
    stage = tile.split("_")[0]; yx = tile.split("_")[1]; d = jd(name) / stage
    a, b = Image.open(d / f"ref_{yx}.png").convert("RGB"), Image.open(d / f"out_{yx}.png").convert("RGB")
    T = a.size[0]
    w = Image.new("RGB", (2 * T, T)); w.paste(a, (0, 0)); w.paste(b, (T, 0)); w.save(jd(name) / f"view_{tile}.png")
    h = T // 2                                                                   # 2x crops of the four quadrants side by side, ref above redraw, for reading small text
    for qi, (cx, cy) in enumerate(((0, 0), (h, 0), (0, h), (h, h))):
        q = Image.new("RGB", (h * 2, h * 2))
        q.paste(a.crop((cx, cy, cx + h, cy + h)).resize((h, h), Image.LANCZOS), (0, 0))
        q.paste(b.crop((cx, cy, cx + h, cy + h)), (h, 0))
        q.crop((0, 0, 2 * h, h)).save(jd(name) / f"view_{tile}_q{qi}.png")
    print(jd(name) / f"view_{tile}.png", "and view_%s_q0..q3.png (each: reference left, redraw right, one quadrant at 1x)" % tile)


def assemble(name, stage):
    d = jd(name) / stage; g = json.loads((d / "grid.json").read_text())
    W, H = g["size"]; T, xs, ys = g["T"], g["xs"], g["ys"]
    acc, wacc = np.zeros((H, W, 3), np.float32), np.zeros((H, W), np.float32); miss = []
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            p = d / f"out_{yi}{xi}.png"
            if not p.exists():
                p = d / f"ref_{yi}{xi}.png"; miss.append(f"{stage}_{yi}{xi}")
            t = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255
            ox0 = (xs[xi - 1] + T - x) if xi else 0; ox1 = (x + T - xs[xi + 1]) if xi + 1 < len(xs) else 0
            oy0 = (ys[yi - 1] + T - y) if yi else 0; oy1 = (y + T - ys[yi + 1]) if yi + 1 < len(ys) else 0
            w = tu.weight(T, T, ox0, ox1, oy0, oy1)
            acc[y:y + T, x:x + T] += t * w[..., None]; wacc[y:y + T, x:x + T] += w
    res = Image.fromarray(((acc / np.maximum(wacc, 1e-6)[..., None]).clip(0, 1) * 255 + .5).astype(np.uint8))
    res.save(d / "stage.png")
    print(f"{stage}/stage.png {res.size}" + (f"; tiles still at the reference: {', '.join(miss)}" if miss else "; all tiles redrawn"))
    return res


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("new"); p.add_argument("name"); p.add_argument("src"); p.add_argument("--hint", default=""); p.add_argument("--sigma-l", type=float, default=4.0); p.add_argument("--sigma-ab", type=float, default=12.0)
    p = sp.add_parser("gen"); p.add_argument("name"); p.add_argument("tile"); p.add_argument("--seed", type=int)
    p = sp.add_parser("all"); p.add_argument("name"); p.add_argument("stage")
    p = sp.add_parser("view"); p.add_argument("name"); p.add_argument("tile")
    p = sp.add_parser("assemble"); p.add_argument("name"); p.add_argument("stage")
    p = sp.add_parser("next"); p.add_argument("name")
    p = sp.add_parser("final"); p.add_argument("name"); p.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "new": new(a)
    elif a.cmd == "gen": gen(a.name, a.tile, a.seed)
    elif a.cmd == "all":
        gr = json.loads((jd(a.name) / a.stage / "grid.json").read_text())
        for yi in range(len(gr["ys"])):
            for xi in range(len(gr["xs"])):
                if not (jd(a.name) / a.stage / f"out_{yi}{xi}.png").exists():
                    gen(a.name, f"{a.stage}_{yi}{xi}")
    elif a.cmd == "view": view(a.name, a.tile)
    elif a.cmd == "assemble": assemble(a.name, a.stage)
    elif a.cmd == "next":
        cut_refs(a.name, "s4", Image.open(jd(a.name) / "s2" / "stage.png").convert("RGB"))
    elif a.cmd == "final":
        assemble(a.name, "s4"); Image.open(jd(a.name) / "s4" / "stage.png").save(a.out); print("wrote", a.out)


if __name__ == "__main__":
    main()
