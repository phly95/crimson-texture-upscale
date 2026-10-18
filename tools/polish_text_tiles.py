"""polish_text_tiles.py NAME UID [--n 8] [--min-words 8]: lettering polish of a finished per-tile-prompt job. For every s4 tile whose prompt quotes at least MIN words: N extra seeds
(tile_best.py keeps the best), then words stitched across the draws (tile_stitch.py); the stage is reassembled (the job's seam_assemble.py + seams_s4.json when present, else tile_job.py
assemble) and submitted as the newest version of the unit."""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tile_job as tj
import tile_best as tb
import tile_stitch as ts
import tile_batch as tbatch

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("uid"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--min-words", type=int, default=8); ap.add_argument("--seed0", type=int, default=300)
a = ap.parse_args()
PY = str(ROOT / "venv/bin/python")
gr = json.loads((tj.jd(a.name) / "s4" / "grid.json").read_text())
for yi in range(len(gr["ys"])):
    for xi in range(len(gr["xs"])):
        tile = f"s4_{yi}{xi}"
        if len(tb.expected(a.name, tile)) < a.min_words:
            continue
        if not list((tj.jd(a.name) / "s4").glob(f"out_{yi}{xi}.s*.png")):
            subprocess.run([PY, "-I", str(ROOT / "tools/tile_best.py"), a.name, tile, "--n", str(a.n), "--seed0", str(a.seed0)], cwd=ROOT)
        ts.stitch(a.name, tile)
out = ROOT / "tile_job" / f"{a.name}_4x_polished.png"
seam = tj.jd(a.name) / "seam_assemble.py"
if seam.exists() and (tj.jd(a.name) / "seams_s4.json").exists():
    subprocess.run([PY, "-I", str(seam), a.name, "s4", str(out)], cwd=ROOT)
else:
    from PIL import Image
    tj.assemble(a.name, "s4"); Image.open(tj.jd(a.name) / "s4" / "stage.png").save(out)
print(a.name, tbatch.submit(a.uid, out, "Tiled FLUX 4x, words stitched across seeds"), flush=True)
