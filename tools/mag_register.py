"""mag_register.py COVER [--label TEXT] [--no-approve]: compose a cover from its parts, add it as the newest custom version of its unit (flux_fix/submit_fix.py: alpha tolerance check) and, unless
--no-approve, append an 'approve' line for that version to pipeline/reviews.jsonl (backup pipeline/reviews.jsonl.bak-pre-parts made once)."""
import argparse, json, re, shutil, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mag_parts as mp

ROOT = mp.ROOT
ap = argparse.ArgumentParser(); ap.add_argument("cover"); ap.add_argument("--label", default="rebuilt from separately upscaled parts"); ap.add_argument("--no-approve", action="store_true")
a = ap.parse_args()
out = mp.PARTS / a.cover / "composed.png"
mp.compose(a.cover, str(out))
uid = mp.UIDS[a.cover]
r = subprocess.run([str(ROOT / "venv/bin/python"), str(ROOT / "flux_fix/submit_fix.py"), uid, str(out), "--label", a.label], capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip()[-300:])
m = re.search(r"submitted as (custom:\d+)", r.stdout)
if not m:
    sys.exit(1 if "outside tolerance" in r.stdout else 0)
if not a.no_approve:
    rv = ROOT / "pipeline" / "reviews.jsonl"
    bak = ROOT / "pipeline" / "reviews.jsonl.bak-pre-parts"
    if not bak.exists():
        shutil.copy(rv, bak)
    with open(rv, "a") as f:
        f.write(json.dumps({"uid": uid, "decision": "approve", "chosen": m.group(1), "seen": [m.group(1)], "issues": [], "text": "", "ts": time.time()}) + "\n")
    print("approved", uid, m.group(1))
