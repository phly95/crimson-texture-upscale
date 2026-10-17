import json, os, shutil, sys
from pathlib import Path

ROOT = Path("/home/deck/workspace/csvm-hd")
PIPE = ROOT / "pipeline"
EXPORT_DIR = ROOT / "winners_export"

def decisions():
    last = {}
    for ln in (PIPE / "reviews.jsonl").read_text().splitlines():
        try:
            r = json.loads(ln)
            last[r["uid"]] = r
        except (ValueError, KeyError):
            continue
    return last

def get_chosen_file(uid, vid):
    d = PIPE / "units" / uid
    if not d.is_dir() or not vid:
        return None
    kind, n = vid.split(":")
    if kind == "custom":
        cjson_path = d / "custom" / "custom.json"
        if cjson_path.exists():
            for it in json.load(open(cjson_path)).get("items", []):
                if it.get("id") == vid:
                    f = d / "custom" / it["file"]
                    if f.exists():
                        return f
        if (d / "custom" / "up1.png").exists():
            return d / "custom" / "up1.png"
        return None

    sub = {"klein": "", "retry": "retry", "magenta": "magenta", "photo": "photo"}.get(kind, "")
    for suf in (("_rgba", "_cm", "") if kind != "magenta" else ("_rgba",)):
        f = d / sub / f"klein_s{n}{suf}.png"
        if f.exists():
            return f
    return None

def run():
    cat = json.load(open(ROOT / "catalog.json"))
    ui_cat = json.load(open(ROOT / "ui_catalog.json"))
    decs = decisions()
    
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    
    (EXPORT_DIR / "world_textures").mkdir(parents=True)
    (EXPORT_DIR / "ui_textures").mkdir(parents=True)
    
    world_copied = 0
    ui_copied = 0
    missing = []
    
    # 1. World textures
    for uid, u in cat["units"].items():
        if u["is_mip"] or u["is_placeholder"]:
            continue
        r = decs.get(uid, {})
        stem = u["stem"]
        
        if r.get("decision") == "approve" and r.get("chosen"):
            src = get_chosen_file(uid, r["chosen"])
            if src and src.exists():
                dst = EXPORT_DIR / "world_textures" / f"{stem}.png"
                shutil.copy2(src, dst)
                world_copied += 1
            else:
                missing.append((uid, stem, r.get("chosen"), "world"))
        else:
            # Keep original
            src = None
            for ch in u["chapters"]:
                p = ROOT / "originals" / ch / f"{stem}.png"
                if p.exists():
                    src = p
                    break
            if not src:
                for ch in u["chapters"]:
                    for s in u["stems"]:
                        p = ROOT / "originals" / ch / f"{s}.png"
                        if p.exists():
                            src = p
                            break
                    if src:
                        break
            if src and src.exists():
                dst = EXPORT_DIR / "world_textures" / f"{stem}.png"
                shutil.copy2(src, dst)
                world_copied += 1
            else:
                missing.append((uid, stem, "original", "world"))
                
    # 2. UI textures
    X = Path(os.path.expanduser("~/Games/CSVM/extracted"))
    for uid, u in ui_cat["units"].items():
        r = decs.get(uid, {})
        rel_src = u.get("ui_src")
        stem = Path(rel_src).stem
        ext = Path(rel_src).suffix
        sub_dir = Path(rel_src).parent
        
        target_dir = EXPORT_DIR / "ui_textures" / sub_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if r.get("decision") == "approve" and r.get("chosen"):
            src = get_chosen_file(uid, r["chosen"])
            if src and src.exists():
                dst = target_dir / f"{stem}.png"
                shutil.copy2(src, dst)
                ui_copied += 1
            else:
                missing.append((uid, stem, r.get("chosen"), "ui"))
        else:
            src = X / rel_src
            if src.exists():
                dst = target_dir / f"{stem}{ext}"
                shutil.copy2(src, dst)
                ui_copied += 1
            else:
                missing.append((uid, stem, "original", "ui"))
                
    print(f"Export finished!")
    print(f"World textures copied: {world_copied}")
    print(f"UI textures copied: {ui_copied}")
    if missing:
        print(f"Missing items ({len(missing)}):", missing[:10])
    else:
        print("Zero missing items!")

if __name__ == "__main__":
    run()
