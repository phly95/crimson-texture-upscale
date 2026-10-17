"""progress.py [-q] [-json]  : how far along the pack is, and which tasks still need work.

Reads tasks/tasks.json plus every out/<id>/gate.json (falling back to the texlab sidecars written by
finish()/mark() when the gate has not been run). A unit counts as DONE when the gate passes it (which
includes an accepted keep_original) or when it is marked esrgan_needed (that is a decision, not a
failure - the maintainer upscaling it is a separate step). Prints per-task lines with -v.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tasks = json.load(open(ROOT / "tasks" / "tasks.json"))
out = ROOT / "out"

report = {}
for t in tasks:
    tid, units = t["id"], t["units"]
    d = out / tid
    g = d / "gate.json"
    st = {}
    if g.exists():
        for uid, r in json.load(open(g)).items():
            st[uid] = "done" if r["pass"] else ("esrgan" if r.get("status") == "esrgan_needed"
                                                else ("keep" if r.get("status") == "keep_original" else "fail"))
    for p in d.glob("*.json"):                      # sidecars, in case the gate was not re-run
        if p.name == "gate.json":
            continue
        m = json.load(open(p))
        uid = m.get("uid")
        if uid and uid not in st:
            s = m.get("status")
            st[uid] = "esrgan" if s == "esrgan_needed" else ("keep" if s == "keep_original" else "made")
    for uid in units:
        st.setdefault(uid, "todo")
    n = len(units)
    c = {k: sum(1 for v in st.values() if v == k) for k in ("done", "keep", "esrgan", "fail", "made", "todo")}
    report[tid] = {"n": n, **c, "units": st, "family": t.get("family") or t.get("kind")}

tot = {k: sum(r[k] for r in report.values()) for k in ("n", "done", "keep", "esrgan", "fail", "made", "todo")}
if "-json" in sys.argv:
    json.dump({"totals": tot, "tasks": report}, sys.stdout, indent=1)
    sys.exit()
print(f"units {tot['n']}: gate-pass {tot['done']}  keep_original {tot['keep']}  esrgan_needed {tot['esrgan']} "
      f"| open: gate-fail {tot['fail']}  made-unchecked {tot['made']}  untouched {tot['todo']}")
todo = [r for r in report.values() if r["fail"] or r["made"] or r["todo"]]
print(f"tasks with work left: {len(todo)} of {len(report)}")
if "-v" in sys.argv:
    for tid, r in sorted(report.items()):
        if r["fail"] or r["made"] or r["todo"]:
            print(f"  {tid} n={r['n']} pass={r['done']} fail={r['fail']} todo={r['todo']} "
                  f"esrgan={r['esrgan']} keep={r['keep']}  {r['family']}")
if "-q" in sys.argv:
    print(" ".join(sorted(tid for tid, r in report.items() if r["fail"] or r["made"] or r["todo"])))
