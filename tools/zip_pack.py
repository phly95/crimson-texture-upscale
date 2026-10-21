"""zip_pack.py RELEASE_DIR OUT_DIR: the two release zips from an export_pack.py folder, stored (the PNGs are already compressed), entries in a fixed order.
The engine registers entries in zip order and a later entry of the same name-without-extension replaces an earlier one, so where export_pack.py shipped both a .png and a .jpg for one
scrapbook item the one listed in ui_default_last.json is written last."""
import json, sys, zipfile
from pathlib import Path


def pack(folder, zpath, default_last=None):
    default_last = default_last or {}
    files = [p for p in folder.rglob("*") if p.is_file()]

    def key(p):
        rel = p.relative_to(folder.parent).as_posix()
        stem = rel.rsplit(".", 1)[0]
        want = default_last.get(stem.split("/", 1)[1] if "/" in stem else stem)
        rank = 1 if want and p.suffix.lower() == "." + want else 0
        return (p.parent.as_posix().lower(), p.stem.lower(), rank, p.name)
    files.sort(key=key)
    dirs = sorted({d for p in files for d in [p.parent, *p.parent.parents] if folder.parent in d.parents or d == folder}, key=lambda d: d.as_posix())
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for d in dirs:
            z.writestr(d.relative_to(folder.parent).as_posix() + "/", b"")
        for p in files:
            z.write(p, p.relative_to(folder.parent).as_posix())
    print(zpath, len(files), "files")


if __name__ == "__main__":
    rel, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    dl = json.loads((rel / "ui_default_last.json").read_text()) if (rel / "ui_default_last.json").exists() else {}
    dl = {k.split("ui_textures/", 1)[-1] if False else k: v for k, v in dl.items()}
    pack(rel / "world_textures", out / "crimson_hd_chosen_world_textures.zip")
    pack(rel / "ui_textures", out / "crimson_hd_chosen_ui_textures.zip", dl)
