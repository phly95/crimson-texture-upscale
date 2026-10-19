"""emblem_gallery.py: static thumbnail page of every emblem / logo / nose-art texture (original vs the version in the pack) -> gallery/site/emblems/ (served at /<token>/emblems/)"""
import sys, re, json, html
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "face_audit")); sys.path.insert(0, str(ROOT / "tools"))
import regen as R, export_winners as ew
OUT = ROOT / "gallery/site/emblems"; OUT.mkdir(parents=True, exist_ok=True)
pat = re.compile(r"bigstink|bighairy|logo|decal|insignia|emblem|badge|crest|zacharyandplane", re.I)
cat = json.load(open(ROOT / "catalog.json"))["units"]
ui = json.load(open(ROOT / "ui_catalog.json"))["units"]
units = [(u, v["stem"]) for u, v in cat.items() if (pat.search(v["stem"]) or re.match(r"^\d\d", v["stem"])) and not v["is_mip"] and not v["stem"].startswith("nosegun")]
units += [(uid, v["stem"]) for uid, v in ui.items() if pat.search(v["stem"]) and uid not in cat]
def thumb(im, path, S=256):
    im = im.convert("RGBA"); k = S / max(im.size); im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.NEAREST if k > 1 and max(im.size) <= 32 else Image.LANCZOS)
    bg = Image.new("RGBA", im.size, (255, 255, 255, 0)); bg.alpha_composite(im); bg.save(path)
rows = []
for uid, stem in sorted(units, key=lambda x: x[1].lower()):
    d = ROOT / "pipeline/units" / uid
    if not (d / "orig.png").exists(): continue
    try:
        cur = R.latest_file(uid) or (ew.get_chosen_file(uid, R.last[uid]["chosen"]) if uid in R.last else None)
    except Exception:
        cur = None
    thumb(Image.open(d / "orig.png"), OUT / f"{uid}_o.png")
    if cur: thumb(Image.open(cur), OUT / f"{uid}_n.png")
    o = Image.open(d / "orig.png"); n = Image.open(cur) if cur else None
    rows.append(dict(uid=uid, stem=stem, o=f"{uid}_o.png", n=f"{uid}_n.png" if cur else None, osz=f"{o.width}x{o.height}", nsz=f"{n.width}x{n.height}" if n else "", ver=(R.last.get(uid, {}).get("chosen") or "")))
# the Fortune Hunters nose-art candidates from this session
Z = ROOT / "face_audit/zach"
extra = []
for name, src in [("48gypsy emblem (reference)", str(ROOT / "face_audit/zach/gypsy_hd.png")), ("Card-fan logo (07fhunter)", "/home/deck/.claude/uploads/8d33977a-9b65-41a7-af0e-5ee75413ed5e/97abb752-image.webp")] + [(f"{lab} (red = your baselines, green = OG oval)", str(Z / fn)) for lab, fn in [("Pointed-tip fit + Flux wrap, seed 1202 (submitted)", "g_tip_s1202.png"), ("Pointed-tip fit + Flux wrap, seed 1201", "g_tip_s1201.png"), ("Pointed-tip fit + Flux wrap, seed 1203", "g_tip_s1203.png"), ("Pointed-tip fit before Flux", "g_tip_in.png"), ("Flux wrap, baselines+oval blend, seed 1100", "g_af35_s1100.png"), ("Flux wrap, baselines+oval blend, seed 1101", "g_af35_s1101.png"), ("Flux wrap, baselines+oval blend, seed 1102", "g_af35_s1102.png"), ("Flux wrap, baselines+oval blend, seed 1103", "g_af35_s1103.png"), ("Flux wrap, baselines only, seed 1001", "g_g31_s1001.png"), ("Flux self-placed, no guides, seed 900", "g_flat_s900.png")]]:
    if Path(src).exists():
        im = Image.open(src).convert("RGBA")
        if "Flux" in name: im = im.crop((40, 60, 700, 800))
        fn = "x_" + re.sub(r"\W+", "_", name) + ".png"; thumb(im, OUT / fn, 384); extra.append((name, fn))
(OUT / "data.json").write_text(json.dumps(dict(rows=rows, extra=extra)))
tpl = (ROOT / "tools/emblem_gallery.html").read_text()
(OUT / "index.html").write_text(tpl)
print(len(rows), "units,", len(extra), "extras ->", OUT)
