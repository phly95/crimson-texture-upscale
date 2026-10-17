"""flipsheet.py TASK_ID [--cols N] [--seeds 1,2] [--cell-scale S]

Animation frames drawn TOGETHER: the frames of one flipbook task are laid out in a grid (each on its green screen, exactly as the pipeline's in.png), klein redraws the
whole sheet in one go so the subject keeps one identity (same face, suit, briefcase) across the frames, and the redrawn cells are cut back out and become the
klein_s<seed>.png of each frame's unit, which then goes through the normal colour match / alpha finalize and the review queue.
Needs the klein server (started here if it is not up); run it while the pipeline is not rendering."""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, "/home/deck/workspace/csvm-hd/tools/pipeline")
import run                                                     # noqa: E402  (functions only; main() is guarded)
from PIL import Image                                          # noqa: E402

PROMPTS = {
    "t0000": ("a sprite sheet of six animation frames of the same man running in side profile, facing right, on a flat pure green (#00FF00) background",
              "short dark brown hair, light skin, a charcoal-grey business suit, a white collared shirt with a dark tie, polished black leather shoes, and a black rectangular leather briefcase carried in one hand"),
    "t0001": ("a sprite sheet of six animation frames of the same man walking in side profile, facing right, on a flat pure green (#00FF00) background",
              "short dark brown hair, light skin, a charcoal-grey business suit, a white collared shirt with a dark tie, polished black leather shoes, and a black rectangular leather briefcase carried in one hand"),
}

ap = argparse.ArgumentParser()
ap.add_argument("task"); ap.add_argument("--cols", type=int, default=3); ap.add_argument("--seeds", default="1,2"); ap.add_argument("--cell-scale", type=float, default=1.0)
a = ap.parse_args()
seeds = [int(s) for s in a.seeds.split(",")]

cfg = run.load_cfg()
uids, idx, tasks = run.universe(cfg)
t = tasks[a.task]
frames = [u["uid"] if isinstance(u, dict) else u for u in t["units"]]
frames = sorted(frames, key=lambda u: idx[u][1]["stem"])        # bmanrun1, bmanrun2, ... in animation order
print("frames:", [idx[u][1]["stem"] for u in frames])

metas = {}
for uid in frames:
    m = run.read_meta(uid)
    if not m or not (run.UN / uid / "in.png").exists():
        m = run.new_meta(uid, idx, tasks, 0)
    metas[uid] = m

cw, ch = Image.open(run.UN / frames[0] / "in.png").size
cw, ch = int(round(cw * a.cell_scale / 16) * 16), int(round(ch * a.cell_scale / 16) * 16)
rows = -(-len(frames) // a.cols)
sheet = Image.new("RGB", (cw * a.cols, ch * rows), (0, 255, 0))
for k, uid in enumerate(frames):
    sheet.paste(Image.open(run.UN / uid / "in.png").convert("RGB").resize((cw, ch), Image.LANCZOS), ((k % a.cols) * cw, (k // a.cols) * ch))
SD = run.PIPE / "flipsheets" / a.task
SD.mkdir(parents=True, exist_ok=True)
sheet.save(SD / "sheet_in.png")
what, who = PROMPTS[a.task]
prompt = (f"Redraw this as a crisp, photorealistic version: {what}, arranged in a grid of {a.cols} columns and {rows} rows, one frame per cell. "
          f"Keep every frame's pose, size and position exactly as in the input and keep the cells aligned; do not add or remove frames, text or shadows. "
          f"It is one single man, who looks exactly the same in every frame: {who}. The same face, hair, clothing and briefcase in every frame; only the pose changes. "
          f"Real fabric texture with natural folds, sharp edges against the green background, no green tint on the figure.")
(SD / "prompt.txt").write_text("PROMPT: " + prompt)

if not run.kserver.up():
    run.kserver.start(run.PIPE / "klein_server.log")
for seed in seeds:
    out = SD / f"sheet_s{seed}.png"
    t0 = time.time()
    run.kserver.gen(SD / "sheet_in.png", out, prompt, sheet.width, sheet.height, seed, timeout=600)
    print(f"seed {seed}: {time.time() - t0:.0f} s -> {out}")
    big = Image.open(out).convert("RGB")
    for k, uid in enumerate(frames):
        d = run.UN / uid
        W, H = Image.open(d / "in.png").size
        cell = big.crop(((k % a.cols) * cw, (k // a.cols) * ch, (k % a.cols + 1) * cw, (k // a.cols + 1) * ch)).resize((W, H), Image.LANCZOS)
        cell.save(d / f"klein_s{seed}.png")

for uid in frames:
    m = metas[uid]; d = run.UN / uid
    (d / "prompt.txt").write_text("PROMPT: " + prompt)
    m["klein"] = {}
    for seed in seeds:
        rec = {"ok": True, "s": 0.0, "attempt": 1, "q": run.score_output(d, seed), "cm": False, "sheet": a.task}
        try:
            run.colormatch(d, seed); rec["cm"] = True; rec["q_cm"] = run.score_output(d, seed, "_cm")
        except Exception as e:                                 # noqa: BLE001
            print(uid, "colormatch failed", repr(e))
        try:
            fr = run.finalize(d, seed)
            if fr:
                rec["rgba"] = fr
        except Exception as e:                                 # noqa: BLE001
            print(uid, "finalize failed", repr(e))
        m["klein"][str(seed)] = rec
    m["best"] = max(m["klein"], key=lambda x: (m["klein"][x].get("q_cm") or m["klein"][x].get("q", {})).get("score", -9))
    m["writer"] = {"s": 0.0, "tokens": 0, "tries": 1, "model": f"hand-written sheet prompt ({a.task}, {len(frames)} frames drawn together)"}
    m["status"] = "done"; m["pv"] = run.PV; m["t_update"] = time.time(); m["sheet"] = a.task
    run.save_meta(m)
print("done")
