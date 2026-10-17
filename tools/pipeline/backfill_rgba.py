"""backfill_rgba.py: write klein_s<seed>_rgba.png (alpha from the original, green screen removed) for every finished unit and record it in meta.json (idempotent)."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from finalize import finalize

UN = Path(__file__).resolve().parents[2] / "pipeline" / "units"
n = done = 0
for d in sorted(UN.iterdir()):
    mp = d / "meta.json"
    if not mp.exists():
        continue
    m = json.loads(mp.read_text())
    if m.get("status") != "done" or m.get("soft"):
        continue
    changed = False
    for sub, recs in ((None, m.get("klein") or {}), ("retry", (m.get("retry") or {}).get("klein") or {})):
        for seed, rec in recs.items():
            if not rec.get("ok") or (rec.get("rgba") and "--force" not in sys.argv):
                continue
            try:
                fr = finalize(d, seed, sub)
            except Exception as e:                       # noqa: BLE001
                print(d.name, seed, "failed", repr(e)[:100]); continue
            if fr:
                rec["rgba"] = fr; changed = True; n += 1
    if changed:
        m2 = json.loads(mp.read_text())                  # re-read just before writing (the pipeline may have touched it)
        for sub, key in ((None, "klein"), ("retry", "retry")):
            src = m.get("klein") if key == "klein" else (m.get("retry") or {}).get("klein")
            dst = m2.get("klein") if key == "klein" else (m2.get("retry") or {}).get("klein")
            for seed, rec in (src or {}).items():
                if rec.get("rgba") and dst and seed in dst:
                    dst[seed]["rgba"] = rec["rgba"]
        tmp = d / "meta.json.tmp"; tmp.write_text(json.dumps(m2)); os.replace(tmp, mp); done += 1
print("finalized", n, "images in", done, "units")
