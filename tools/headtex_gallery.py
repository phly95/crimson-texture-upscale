"""headtex_gallery.py: static page of the character textures that wrap around 3D heads/bodies (original vs the version in the pack) -> gallery/site/headtex/ (served at /<token>/headtex/)"""
import sys, json
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "face_audit")); sys.path.insert(0, str(ROOT / "tools"))
import regen as R, export_winners as ew
OUT = ROOT / "gallery/site/headtex"; OUT.mkdir(parents=True, exist_ok=True)
cat = json.load(open(ROOT / "catalog.json"))["units"]
SECTIONS = [
    ("Head wraps", "UV maps for the 3D heads: profile and front of a head laid out flat (hair, ear, goggles, face).", "cphead cphead02 womanhead womanface fbhead1"),
    ("Body and limb skins", "Flight suits, torsos, arms and legs of the same figures.", "cpsuit128 cpsuit128_01 cpsuit128_02 womanchest womanarm woman1 fbsuit128_01 fbsuit128_02"),
    ("Passenger figures", "Photo portraits on the passenger meshes (p_jack, p_pick, p_ilsa...). Flat cut-outs, not a wrap, but they are the faces seen in 3D.", "pass_bswan pass_fas pass_jack pass_john pass_pick pass_sparks pass_waldo ilsafas uboothe pass_coat1 pass_coat2 sweater"),
]
byst = {v["stem"]: (u, v) for u, v in cat.items() if not v.get("is_mip")}
def thumb(im, path, S):
    im = im.convert("RGBA"); k = S / max(im.size)
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.NEAREST if k > 1 and max(im.size) <= 128 else Image.LANCZOS)
    im.save(path)
secs = []
for title, blurb, stems in SECTIONS:
    rows = []
    for st in stems.split():
        if st not in byst: continue
        uid, v = byst[st]; d = ROOT / "pipeline/units" / uid
        if not (d / "orig.png").exists(): continue
        try: cur = R.latest_file(uid) or (ew.get_chosen_file(uid, R.last[uid]["chosen"]) if uid in R.last else None)
        except Exception: cur = None
        o = Image.open(d / "orig.png"); n = Image.open(cur) if cur else None
        thumb(o, OUT / f"{uid}_o.png", 320)
        if cur: thumb(n, OUT / f"{uid}_n.png", 640)
        meta = json.load(open(d / "meta.json"))
        rows.append(dict(stem=st, o=f"{uid}_o.png", n=f"{uid}_n.png" if cur else None, osz=f"{o.width}x{o.height}", nsz=f"{n.width}x{n.height}" if n else "",
                         nodes=", ".join((meta.get("nodes") or [])[:5]), ch=", ".join(v["chapters"]) if len(v["chapters"]) < 8 else "all chapters"))
    secs.append(dict(title=title, blurb=blurb, rows=rows))
(OUT / "data.json").write_text(json.dumps(dict(secs=secs)))
(OUT / "index.html").write_text((ROOT / "tools/headtex_gallery.html").read_text())
print(sum(len(s["rows"]) for s in secs), "textures ->", OUT)
