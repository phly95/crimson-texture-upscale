"""register.py: add the coherent water frames out/<group>/wtr000NN.png as the newest custom version of each frame unit (review shows them first).
Frames 1..15 of every group are parked in pipeline/hidden.json so the review only shows frame 0 of each group (the frame that matters most); apply_group.py releases
and approves the other frames once frame 0 is approved. Idempotent (sha1)."""
import hashlib, json, shutil, time
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
P = HERE.parent / "pipeline"
G = json.load(open(HERE / "groups.json"))
hid = json.load(open(P / "hidden.json"))
park = set(hid["uids"]); added = 0
for g, info in G.items():
    for s, f in info["frames"].items():
        src = HERE / "out" / g / f"{s}.png"
        if not src.exists():
            continue
        uid = f["uid"]; cd = P / "units" / uid / "custom"; cd.mkdir(exist_ok=True)
        jp = cd / "custom.json"; items = json.loads(jp.read_text()).get("items", []) if jp.exists() else []
        im = Image.open(src).convert("RGBA")
        tmp = HERE / "out" / g / f"_reg_{s}.png"; im.save(tmp)
        h = hashlib.sha1(tmp.read_bytes()).hexdigest()
        if not any(i.get("sha1") == h for i in items):
            n = max([int(i["id"].split(":")[1]) for i in items] + [0]) + 1
            shutil.copy(tmp, cd / f"up{n}.png")
            items.append({"id": f"custom:{n}", "file": f"up{n}.png", "rgba": False, "alpha": "opaque", "size": list(im.size), "ts": time.time(), "replaces": f.get("chosen") or "klein:1",
                          "sha1": h, "label": f"Coherent water set ({g}), 2D edit"})
            jp.write_text(json.dumps({"items": items}, indent=2)); added += 1
        tmp.unlink()
        if s != "wtr00000" and uid not in {gg["frames"]["wtr00000"]["uid"] for gg in G.values()}:
            park.add(uid)
hid["uids"] = sorted(park); hid["reason"] += "; water frames 1-15 parked until frame 0 of their group is approved (water_job/apply_group.py)"
json.dump(hid, open(P / "hidden.json", "w"), indent=1)
print("added", added, "versions; parked", len(park))
