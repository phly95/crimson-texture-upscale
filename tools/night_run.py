"""night_run.py: everything that is left, one after another (the GPU takes one job at a time):
  1 material-only batch (tile_batch)  2 lettered pictures (tile_text_batch)  3 hangar frames  4 PC_PLANETEMP   (maps: per-tile prompt agent, not here)
Aliases: pictures that are the same art under two names get the same result submitted for both uids. Each step is resumable; a failure only logs."""
import json, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / "venv/bin/python")
sys.path.insert(0, str(ROOT / "tools"))
LIST = ROOT / "tile_job" / "batch1_list.json"


def run(args, log):
    print(time.strftime("%F %T"), "RUN", " ".join(args), flush=True)
    with open(ROOT / "tile_job" / log, "a") as lf:
        subprocess.run([PY, "-I"] + args, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)


def wait_batch_a():
    while subprocess.run(["pgrep", "-f", "tile_batch.py tile_job/batch1_list.json"], capture_output=True).stdout.strip():
        time.sleep(30)


def alias(src_idx, dst_idx, stem_src):
    import tile_batch as tb
    items = json.load(open(LIST))
    fs = sorted((ROOT / "tile_job").glob(f"{stem_src}*_4x_text.png"))
    f = fs[0] if fs else None
    if f:
        print("alias", items[dst_idx][1], tb.submit(items[dst_idx][0], f, "Tiled FLUX 4x, text layer restored (same art as another file)"), flush=True)


if __name__ == "__main__":
    wait_batch_a()
    B = [12, 8, 7, 24, 0, 22, 18, 25, 4] + [31, 32, 33, 34, 37, 38, 42, 43, 44, 50, 51, 52, 53, 26, 27]
    run(["tools/tile_text_batch.py", "tile_job/batch1_list.json", ",".join(map(str, B))], "night_text.log")
    alias(12, 3, "MM_SPLASHBACKGROUND")
    alias(25, 30, "SB_BG_BZOOMNEWS")
    run(["tools/tile_hangar.py"], "night_hangar.log")
    run(["tools/tile_batch.py", "tile_job/batch1_list.json", "19"], "night_misc.log")
    print(time.strftime("%F %T"), "ALL DONE", flush=True)
