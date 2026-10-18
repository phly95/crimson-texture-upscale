"""add_sprites.py GROUP STEM [STEM ...]: add units to the agent's work list (sprites.json order + sprites/<stem>/ workspace + checksum guard).
GROUP is a label stored in meta ('effects', 'structures'); structures get iou_min 0.70 (thin 1-2 px geometry cannot reach 0.85 after a 8x reduction)."""
import hashlib, json, shutil, sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import build_pack as bp

HERE = Path(__file__).resolve().parent
group, stems = sys.argv[1], sys.argv[2:]
cat = json.load(open(bp.ROOT / "catalog.json")); decs = bp.decisions()
S = json.loads((HERE / "sprites.json").read_text())
by = {u["stem"]: uid for uid, u in cat["units"].items() if not u["is_mip"]}
ui = json.load(open(bp.ROOT / "ui_catalog.json"))["units"]
for uid, u in ui.items():                       # UI art (gfx_/rimage_/scrap_ names): by the file stem, after the world textures
    by.setdefault(u["ui_src"].split("/")[-1].rsplit(".", 1)[0], uid)
by.update({uid: uid for uid in ui})             # or by the unit id itself
cat["units"].update({uid: dict(u, is_mip=False, chapters=[], target=u.get("target") or [u["w"] * 5 if "w" in u else 0, 0]) for uid, u in ui.items() if uid not in cat["units"]})
for s in stems:
    uid = by[s]; u = cat["units"][uid]; d = HERE / "sprites" / s; d.mkdir(parents=True, exist_ok=True)
    shutil.copy(bp.ROOT / "pipeline" / "units" / uid / "orig.png", d / "orig.png")
    o = Image.open(d / "orig.png").convert("RGBA"); W, H = u["target"]
    o.resize((W, H), Image.LANCZOS).save(d / "ref_upscale.png")
    r = decs.get(uid, {}); src = bp.version_file(uid, r["chosen"]) if r.get("chosen") else None
    if src:
        shutil.copy(bp.conform(uid, src, cat)[0], d / "prev_redraw.png")
    m = {"uid": uid, "orig_size": list(o.size), "target": [W, H], "chapters": u["chapters"], "group": group}
    if group == "structures":
        m["iou_min"] = 0.70
    json.dump(m, open(d / "meta.json", "w"))
    S["sprites"][s] = m
    if s not in S["order"]:
        S["order"].append(s)
    print("added", s, o.size, "->", (W, H))
json.dump(S, open(HERE / "sprites.json", "w"), indent=1)
G = json.loads((HERE / "guard.json").read_text())
h = lambda p: hashlib.sha1(Path(p).read_bytes()).hexdigest()
for s in S["order"]:
    G["orig"][s] = h(HERE / "sprites" / s / "orig.png")
json.dump(G, open(HERE / "guard.json", "w"))
