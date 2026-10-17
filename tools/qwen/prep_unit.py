"""prep_unit.py UID OUTDIR  -> OUTDIR/UID_orig.png (original), UID_in.png (1024px input for klein/VLM; cut-outs on #00FF00), UID_hint.txt
Importable: prep(uid, outdir) -> (in_png, orig_png, hint)."""
import glob, json, sys
from pathlib import Path
from PIL import Image
ROOT = Path("/home/deck/workspace/csvm-hd")
_idx = None

def find(uid):
    global _idx
    if _idx is None:
        _idx = {}
        for tj in sorted(glob.glob(str(ROOT / "tasks/t*/task.json"))):
            t = json.load(open(tj))
            for u in t.get("units", []):
                _idx[u["uid"]] = (Path(tj).parent, u)
    return _idx[uid]

def prep(uid, outdir):
    d, u = find(uid); out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    im = Image.open(d / "orig" / f"{uid}.png").convert("RGBA"); im.save(out / f"{uid}_orig.png")
    cut = u.get("has_alpha") and (im.getchannel("A").getextrema()[0] < 128)
    if cut:
        bg = Image.new("RGBA", im.size, (0, 255, 0, 255)); bg.alpha_composite(im); base = bg.convert("RGB")
    else:
        base = im.convert("RGB")
    w, h = im.size; k = 1024 / max(w, h)
    nw, nh = max(16, round(w * k / 16) * 16), max(16, round(h * k / 16) * 16)
    base.resize((nw, nh), Image.BICUBIC).save(out / f"{uid}_in.png")
    us = u.get("usage") or {}
    bits = [f"file name '{u['stem']}'", f"{im.size[0]}x{im.size[1]} px, " + ("cut-out shape with an empty background" if cut else "opaque")]
    if us.get("nodes"): bits.append("used on world objects named " + ", ".join(us["nodes"][:6]))
    sf = us.get("surface") or {}
    if sf:
        k = max(sf, key=sf.get)
        if sf[k] > 0.5: bits.append({"ground_up": "mostly on ground surfaces (facing up)", "ceiling_down": "mostly on ceilings (facing down)", "wall_side": "mostly on vertical wall surfaces"}[k])
    if us.get("uv_repeat_median"): bits.append(f"repeats about {us['uv_repeat_median'][0]:g}x{us['uv_repeat_median'][1]:g} per polygon")
    if u.get("flipbook_id"): bits.append("one frame of an animation")
    if u.get("also_named"): bits.append("also named " + ", ".join(u["also_named"][:3]))
    if u.get("ui_note"): bits[0:2] = [f"file name '{u['stem']}'", f"{im.size[0]}x{im.size[1]} px, " + ("with transparency" if cut else "opaque"), u["ui_note"]]
    hint = "; ".join(bits)
    (out / f"{uid}_hint.txt").write_text(hint)
    return out / f"{uid}_in.png", out / f"{uid}_orig.png", hint

if __name__ == "__main__":
    a, o, h = prep(sys.argv[1], sys.argv[2]); print(a, o, h)
