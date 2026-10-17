"""cmp_klein.py OUTDIR : pause the pipeline, stop Strata, klein-render OUTDIR/*/gemini_prompt.txt (seed 1), resume.  Skips finished ones."""
import json, os, re, subprocess, sys, time
from pathlib import Path
from PIL import Image
R = Path("/home/deck/workspace/csvm-hd"); PIPE = R / "pipeline"; out = Path(sys.argv[1])
(PIPE / "PAUSE").touch()
try:
    for _ in range(600):
        if json.load(open(PIPE / "state.json")).get("phase") == "paused":
            break
        time.sleep(5)
    subprocess.run(["podman", "stop", "llm-strata"], capture_output=True, timeout=120)
    time.sleep(8)
    for d in sorted(out.iterdir()):
        g = d / "gemini_prompt.txt"
        if not g.exists() or (d / "gemini_s1.png").exists():
            continue
        m = re.search(r"PROMPT:\s*(.+)", g.read_text(), re.S)
        if not m:
            continue
        W, H = Image.open(d / "in.png").size
        t = time.time()
        rc = subprocess.run([str(R / "tools/qwen/klein.sh"), "9b", str(d / "in.png"), str(d / "gemini_s1.png"), m.group(1).strip()],
                            env=dict(os.environ, W=str(W), H=str(H), SEED="1"), capture_output=True, timeout=400).returncode
        print(f"{d.name}: rc {rc} {time.time()-t:.0f}s", flush=True)
finally:
    (PIPE / "PAUSE").unlink(missing_ok=True)
print("klein comparison done", flush=True)
