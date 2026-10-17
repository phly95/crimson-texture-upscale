"""backfill_cm.py: colour-match every finished unit that has none yet and update meta.json (scores, best seed). Run while the pipeline is stopped."""
import json, sys, importlib.util
from pathlib import Path
ROOT = Path("/home/deck/workspace/csvm-hd"); sys.path.insert(0, str(ROOT / "tools" / "pipeline"))
spec = importlib.util.spec_from_file_location("run", ROOT / "tools/pipeline/run.py"); run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
n = 0
for mp in sorted((ROOT / "pipeline/units").glob("*/meta.json")):
    m = json.load(open(mp)); d = mp.parent
    if m.get("status") != "done": continue
    ch = False
    for s, v in m["klein"].items():
        if v.get("ok") and not v.get("cm") and (d / f"klein_s{s}.png").exists():
            run.colormatch(d, int(s)); v["cm"] = True; v["q_cm"] = run.score_output(d, int(s), "_cm"); ch = True
    if ch:
        ok = [s for s, v in m["klein"].items() if v.get("ok")]
        m["best"] = max(ok, key=lambda x: (m["klein"][x].get("q_cm") or m["klein"][x].get("q", {})).get("score", -9))
        json.dump(m, open(mp, "w")); n += 1
print("backfilled", n, "units")
