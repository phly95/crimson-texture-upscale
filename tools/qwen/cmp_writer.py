"""cmp_writer.py OUTDIR STEM... : run the cloud writer (Gemini via the 9router endpoint) on the FIRST-PASS inputs of the given units
(archive/feedback/<uid> if it exists = the version the user rejected, else units/<uid>) and save OUTDIR/<stem>/{gemini_prompt.txt,qwen_prompt.txt,in.png}."""
import json, os, re, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
R = Path("/home/deck/workspace/csvm-hd"); PIPE = R / "pipeline"
out = Path(sys.argv[1]); stems = sys.argv[2:]
env = dict(os.environ, VLM_URL="http://100.117.72.11:20128/v1/chat/completions", VLM_MODEL=os.environ.get("GEM_MODEL", "ag/gemini-3.8-flash-medium"),
           VLM_EXTRA="{}", VLM_KEY_FILE=str(R / "gemini_api.key"))


def one(stem):
    src = next(iter(sorted((PIPE / "archive" / "feedback").glob(stem + "__*"))), None) or next(iter(sorted((PIPE / "units").glob(stem + "__*"))))
    d = out / stem; d.mkdir(parents=True, exist_ok=True)
    m = json.loads((src / "meta.json").read_text())
    orig = next(p for p in (src / "orig_lift.png", src / "orig_up.png", src / "orig.png") if p.exists())
    for f in ("in.png", "prompt.txt", "klein_s1.png", "klein_s1_cm.png", "klein_s2.png", "digest.txt"):
        if (src / f).exists():
            shutil.copy(src / f, d / ({"prompt.txt": "qwen_prompt.txt", "klein_s1.png": "qwen_s1.png", "klein_s1_cm.png": "qwen_s1_cm.png", "klein_s2.png": "qwen_s2.png"}.get(f, f)))
    shutil.copy(orig, d / "orig.png")
    t = time.time()
    p = subprocess.run([sys.executable, "-I", str(R / "tools/qwen/vlm_prompt.py"), str(d / "in.png"), str(orig), m["hint"]], env=env, capture_output=True, text=True, timeout=900)
    (d / "gemini_prompt.txt").write_text(p.stdout); (d / "gemini.err").write_text(p.stderr)
    ok = bool(re.search(r"PROMPT:\s*(.+)", p.stdout, re.S))
    print(f"{stem}: rc {p.returncode} ok {ok} {time.time()-t:.0f}s {p.stderr.strip()[-120:]}", flush=True)


with ThreadPoolExecutor(int(os.environ.get("PAR", "5"))) as ex:
    list(ex.map(one, stems))
