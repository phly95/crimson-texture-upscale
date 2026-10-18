"""flux_gen.py STEM ATTEMPT --prompt-file P [--guide G.png] [--seeds 1,2,3]: one guided FLUX klein redraw of a unit, processed like the pipeline does
(undo brighten/rotate, colour match, green-screen key -> alpha), into flux_fix/work/<STEM>/<ATTEMPT>/ (klein_sN.png, klein_sN_cm.png, klein_sN_rgba.png).
--guide replaces the picture FLUX edits (default: the unit's own in.png, i.e. the original on a green screen, brightened/rotated if the unit needs it): it must be
in the SAME space as in.png (same rotation/brightening, same aspect); it is resized to in.png's size. The prompt is plain text (the pipeline's 'PROMPT:' prefix is added).
Prints the output paths and the alpha tolerance metrics of each seed. Starts the klein server if needed and stops it afterwards unless the pipeline is running.
Never touches pipeline/ unit folders and never uses run.save_meta."""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools" / "pipeline")); sys.path.insert(0, str(HERE))
import run                                                                       # noqa: E402
import tol, alpha_tol                                                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("stem"); ap.add_argument("attempt"); ap.add_argument("--prompt-file", required=True)
ap.add_argument("--guide"); ap.add_argument("--seeds", default="1,2,3")
a = ap.parse_args()
uid = alpha_tol.unit_of(a.stem)
U = run.UN / uid
w = HERE / "work" / a.stem / a.attempt
w.mkdir(parents=True, exist_ok=True)
for f in ("orig.png", "orig_lift.png", "hint.txt"):
    if (U / f).exists():
        shutil.copy(U / f, w / f)
m = json.loads((U / "meta.json").read_text())
W, H = Image.open(U / "in.png").size
(Image.open(a.guide) if a.guide else Image.open(U / "in.png")).convert("RGB").resize((W, H), Image.LANCZOS).save(w / "in.png")
prompt = Path(a.prompt_file).read_text().replace("PROMPT:", "", 1).strip()
(w / "prompt.txt").write_text("PROMPT: " + prompt)
started = False
if not run.kserver.up():
    run.kserver.start(run.PIPE / "klein_server.log"); started = True
res = {}
for seed in [int(s) for s in a.seeds.split(",")]:
    out = w / f"klein_s{seed}.png"
    t0 = time.time()
    run.kserver.gen(w / "in.png", out, prompt, W, H, seed, timeout=600)
    if m.get("lift"):
        Image.open(out).save(w / f"klein_s{seed}_lifted.png")
        run.apply_gamma(Image.open(out), 1.0 / m["lift"]).save(out)
    if m.get("rot"):
        Image.open(out).save(w / f"klein_s{seed}_up.png")
        Image.open(out).rotate(m["rot"], expand=True).save(out)
    try:
        run.colormatch(w, seed)
    except Exception as e:                                                       # noqa: BLE001
        print("colormatch failed", repr(e))
    try:
        run.finalize(w, seed)
    except Exception as e:                                                       # noqa: BLE001
        print("finalize failed", repr(e))
    f = next((w / n for n in (f"klein_s{seed}_rgba.png", f"klein_s{seed}_cm.png", f"klein_s{seed}.png") if (w / n).exists()), None)
    res[seed] = (f, tol.metrics(U / "orig.png", f) if f else None)
    print(f"seed {seed} ({time.time() - t0:.0f}s): {f}  {json.dumps(res[seed][1])}")
if started and not subprocess.run(["pgrep", "-f", "tools/pipeline/run.py"], capture_output=True).stdout.strip():
    run.kserver.stop()
