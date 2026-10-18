"""prep_buttons.py: build the work list of lettered 4-state capsule buttons (sprites.json, sprites/<name>/, guard.json)."""
import hashlib, json, re, shutil, sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import build_pack as bp
HERE = Path(__file__).resolve().parent
ui = json.load(open(bp.ROOT / "ui_catalog.json")); decs = bp.decisions()
SKIP = re.compile(r"SCROLL|ARROW|CHECKBOX|RADIO|SLIDER|PAPERBUTTON|ICON|BUTTONS|KEYSANDBUTTONS|CHECK$|_TAB$|TAB$|LONGTAB|^MP_B_(LARGE|MEDIUM|SMALL)$|OKCANCEL")
order, sprites, G = [], {}, {"orig": {}, "verify": hashlib.sha1((HERE.parent / "tools/agents/sprite_verify.py").read_bytes()).hexdigest()}
for uid, u in sorted(ui["units"].items(), key=lambda x: x[1]["ui_src"]):
    n = u["ui_src"].split("/")[-1].rsplit(".", 1)[0]
    if not re.match(r"[A-Z]{2}_B_", n) or SKIP.search(n):
        continue
    d = HERE / "sprites" / n; d.mkdir(parents=True, exist_ok=True)
    shutil.copy(bp.ROOT / "pipeline/units" / uid / "orig.png", d / "orig.png")
    o = Image.open(d / "orig.png").convert("RGBA"); W, H = u["target"] if u.get("target") else (o.width * 8, o.height * 8)
    r = decs.get(uid, {}); src = bp.version_file(uid, r["chosen"]) if r.get("chosen") else None
    tw, th = o.width * 5, o.height * 5                      # 5x like the other UI art (about 1000 px wide): the engine fits it to the original size
    o.resize((tw, th), Image.LANCZOS).save(d / "ref_upscale.png")
    if src:
        Image.open(src).convert("RGBA").resize((tw, th), Image.LANCZOS).save(d / "prev_redraw.png")
    m = {"uid": uid, "orig_size": list(o.size), "target": [tw, th], "chapters": [], "group": "buttons", "states": 4, "state_h": o.height // 4,
         "label": re.sub(r"^[A-Z]{2}_B_", "", n), "iou_min": 0.9, "de_max": 10, "ui_src": u["ui_src"]}
    json.dump(m, open(d / "meta.json", "w")); sprites[n] = m; order.append(n)
    G["orig"][n] = hashlib.sha1((d / "orig.png").read_bytes()).hexdigest()
json.dump({"order": order, "sprites": sprites}, open(HERE / "sprites.json", "w"), indent=1)
json.dump(G, open(HERE / "guard.json", "w"))
print(len(order), "buttons")
