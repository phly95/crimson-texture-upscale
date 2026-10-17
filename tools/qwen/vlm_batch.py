"""vlm_batch.py OUTDIR PAR UID [UID ...]   writes OUTDIR/UID_prompt.txt (ANALYSIS+PROMPT) with PAR requests in flight against Strata :9931.
Prints per-unit wall time and the aggregate: completion tokens, images/min, tok/s."""
import subprocess, sys, time, json
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/home/deck/workspace/csvm-hd/tools/qwen")
from prep_unit import prep
out, par, uids = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
PY = "/home/deck/workspace/csvm-hd/venv/bin/python"

def one(uid):
    t = time.time()
    inp, orig, hint = prep(uid, out)
    r = subprocess.run([PY, "-I", "/home/deck/workspace/csvm-hd/tools/qwen/vlm_prompt.py", str(inp), str(orig), hint], capture_output=True, text=True)
    (open(f"{out}/{uid}_prompt.txt", "w")).write(r.stdout)
    toks = 0
    for ln in r.stderr.splitlines():
        if "completion_tokens" in ln:
            toks = int(ln.split("'completion_tokens': ")[1].split(",")[0])
    return uid, time.time() - t, toks, ("PROMPT:" in r.stdout)

from prep_unit import find
for u in uids: find(u)   # build the unit index once, before the threads start
t0 = time.time()
with ThreadPoolExecutor(par) as ex: res = list(ex.map(one, uids))
wall = time.time() - t0
for u, s, k, ok in res: print(f"{u}: {s:.0f}s, {k} completion tokens, {'ok' if ok else 'NO PROMPT'}")
tot = sum(k for _, _, k, _ in res)
print(f"AGGREGATE par={par}: {len(res)} units in {wall:.0f}s = {60*len(res)/wall:.2f} images/min, {tot} tokens, {tot/wall:.1f} tok/s")
