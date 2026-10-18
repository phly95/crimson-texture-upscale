"""apply_group.py: once frame 0 of a water group is approved on a coherent-set version (custom:N), approve the same set's other frames and un-park them.
Frames 8..15 share unit ids with 0..7, so one decision per unit covers both. Idempotent."""
import json, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE.parent / "pipeline"
G = json.load(open(HERE / "groups.json"))
last = {}
for ln in (P / "reviews.jsonl").read_text().splitlines():
    try:
        r = json.loads(ln); last[r["uid"]] = r
    except ValueError:
        pass
hid = json.load(open(P / "hidden.json")); park = set(hid["uids"]); out = []
for g, info in G.items():
    f0 = last.get(info["frames"]["wtr00000"]["uid"])
    if not f0 or f0["decision"] != "approve" or not str(f0.get("chosen", "")).startswith("custom:"):
        print(g, "frame 0 not approved on a custom version; skipped"); continue
    for s, f in info["frames"].items():
        uid = f["uid"]
        if uid == info["frames"]["wtr00000"]["uid"]:
            continue
        items = json.loads((P / "units" / uid / "custom" / "custom.json").read_text())["items"]
        mine = [i["id"] for i in items if g in i.get("label", "")]
        if not mine:
            print("no coherent version for", uid); continue
        ch = mine[-1]
        if (last.get(uid) or {}).get("chosen") != ch or last[uid]["decision"] != "approve":
            out.append(json.dumps({"uid": uid, "decision": "approve", "chosen": ch, "seen": [ch], "issues": [], "text": "water set approved with frame 0", "ts": time.time()}))
        park.discard(uid)
if out:
    with open(P / "reviews.jsonl", "a") as fh:
        fh.write("\n".join(out) + "\n")
hid["uids"] = sorted(park); json.dump(hid, open(P / "hidden.json", "w"), indent=1)
print("approved", len(out), "frame units; still parked", len(park))
