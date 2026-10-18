#!/usr/bin/env python3
"""supervisor.py: keeps the FLUX pipeline 'alive' for the review page without holding the GPU while idle.

run.py ends (and frees the GPU) when it has nothing to do, so a later thumbs-down would never be processed. This service waits for work and starts run.py
for it: a new review reject (FLUX route; agent-routed ones are not ours), a 'redraw' or 'photo' request, a magenta redo, or a changed config 'redo' token.
run.py then does everything pending and exits again ('run ended; GPU free'). Stdlib only. systemd user service csvm-pipeline-supervisor."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / "pipeline"
REVIEWS, MAGENTA, CONFIG = PIPE / "reviews.jsonl", PIPE / "magenta_requests.jsonl", PIPE / "config.json"
OUT = PIPE / "run.out"


def log(msg):
    with open(PIPE / "run.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} supervisor: {msg}\n")


def running():
    r = subprocess.run(["pgrep", "-f", "tools/pipeline/run.py"], capture_output=True, text=True)
    return [p for p in r.stdout.split() if int(p) != os.getpid()]


def size(p):
    try:
        return p.stat().st_size
    except OSError:
        return 0


def redo_tokens():
    try:
        return json.dumps(json.loads(CONFIG.read_text()).get("redo") or {}, sort_keys=True)
    except (OSError, ValueError):
        return ""


def main():
    while running():                                           # an existing run (started by hand) finishes first
        time.sleep(15)
    off_rev, off_mag, tok = size(REVIEWS), size(MAGENTA), redo_tokens()
    log("watching for review work (idle = GPU free)")
    while True:
        work = False
        n = size(REVIEWS)
        if n > off_rev:
            with open(REVIEWS) as f:
                f.seek(off_rev)
                for ln in f.read().splitlines():
                    try:
                        r = json.loads(ln)
                    except ValueError:
                        continue
                    if (r.get("decision") == "reject" and r.get("route") != "agent") or r.get("decision") in ("redraw", "photo"):
                        work = True
            off_rev = n
        if size(MAGENTA) > off_mag:
            off_mag, work = size(MAGENTA), True
        t = redo_tokens()
        if t != tok:
            tok, work = t, True
        if work and not running():
            log("review work arrived: starting run.py")
            with open(OUT, "ab") as out:
                subprocess.run([str(ROOT / "venv" / "bin" / "python"), "-I", str(ROOT / "tools" / "pipeline" / "run.py")], stdout=out, stderr=subprocess.STDOUT, cwd=ROOT)
            log("run.py ended; GPU free")
            off_rev, off_mag = size(REVIEWS), size(MAGENTA)    # anything that arrived during the run was handled by it (it re-reads requests in its loop)
        time.sleep(15)


if __name__ == "__main__":
    main()
