"""Seam-based assemble: instead of feathering across the whole tile overlap (which ghosts text drawn differently by two tiles),
switch tiles along a narrow ramp placed at a chosen row/column (pick rows/cols that hold no text).
usage: seam_assemble.py NAME STAGE OUT.png  (reads tile_job/NAME/seams_STAGE.json:
  {"y": [{"x0":0,"x1":415,"seams":[[596,16]]}, {"x0":415,"x1":99999,"seams":[[596,80]]}], "x": [[790,60]]}
 seams = [[position, ramp_width], ...], one per adjacent tile pair in order; y seams may differ per x zone)"""
import json, sys
import numpy as np
from PIL import Image
from pathlib import Path
name, stage, out = sys.argv[1:4]
d = Path("/home/deck/workspace/csvm-hd/tile_job") / name / stage
g = json.loads((d / "grid.json").read_text()); W, H = g["size"]; T, xs, ys = g["T"], g["xs"], g["ys"]
cfg = json.loads((d.parent / f"seams_{stage}.json").read_text())

def w1d(L, n, seams):
    """(n, L) partition-of-unity weights: tile i is replaced by tile i+1 across seams[i] = [position, ramp_width]"""
    p = np.arange(L, dtype=np.float32); step = []
    for pos, r in seams:
        t = np.clip((p - (pos - r / 2)) / max(r, 1e-3), 0, 1); step.append(t * t * (3 - 2 * t))
    S = [np.ones(L, np.float32)] + step + [np.zeros(L, np.float32)]   # S[i] = how much of tile i-1 has handed over
    return np.stack([S[i] - S[i + 1] for i in range(n)])

tiles = {(yi, xi): np.asarray(Image.open(d / f"out_{yi}{xi}.png").convert("RGB")).astype(np.float32) for yi in range(len(ys)) for xi in range(len(xs))}
wx = w1d(W, len(xs), cfg["x"])
res = np.zeros((H, W, 3), np.float32); wsum = np.zeros((H, W), np.float32)
TRIM = 0   # FLUX draws a faint dark line along the outer edge of a tile: never use the outermost TRIM px of a tile that has a neighbour
for z in cfg["y"]:
    x0, x1 = z["x0"], min(z["x1"], W)
    wy = w1d(H, len(ys), z["seams"])
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            wgt = wy[yi][:, None] * wx[xi][None, x0:x1]
            ya, yb = y, y + T; xa, xb = max(x, x0), min(x + T, x1)
            if xb <= xa: continue
            sub = wgt[ya:yb, xa - x0:xb - x0].copy()
            if xi > 0: sub[:, :max(0, x + TRIM - xa)] = 0
            if xi < len(xs) - 1: sub[:, max(0, x + T - TRIM - xa):] = 0
            if yi > 0: sub[:max(0, y + TRIM - ya), :] = 0
            if yi < len(ys) - 1: sub[max(0, y + T - TRIM - ya):, :] = 0
            wsum[ya:yb, xa:xb] += sub
            res[ya:yb, xa:xb] += tiles[(yi, xi)][:, xa - x:xb - x] * sub[..., None]
res = res / np.maximum(wsum, 1e-3)[..., None]
Image.fromarray(res.clip(0, 255).astype(np.uint8)).save(out); print("wrote", out, res.shape)
