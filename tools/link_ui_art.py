"""link_ui_art.py: link or copy upgraded UI textures into CSVM extracted/ tree.
Safe: creates backups of original files (*.orig_bak) so nothing is destructively overwritten.
"""
import json, os, shutil, sys
from pathlib import Path

ROOT = Path("/home/deck/workspace/csvm-hd")
GAME = Path(os.path.expanduser("~/Games/CSVM/extracted"))
WINNERS_UI = ROOT / "winners_export" / "ui_textures"
UI_CAT = json.load(open(ROOT / "ui_catalog.json"))
REVIEWS = {}
for line in open(ROOT / "pipeline/reviews.jsonl"):
    try:
        r = json.loads(line)
        REVIEWS[r["uid"]] = r
    except:
        pass

def install_ui():
    count = 0
    skipped_orig = 0
    for uid, u in UI_CAT["units"].items():
        r = REVIEWS.get(uid, {})
        # Only install approved upgrades
        if r.get("decision") != "approve":
            skipped_orig += 1
            continue
            
        rel_src = u["ui_src"] # e.g. "rimage/rm-m1map.png" or "rof/ASSETS/GRAPHICS/FC_B_PAPERBUTTON.PNG"
        target_file = GAME / rel_src
        source_winner = WINNERS_UI / rel_src
        
        # In case the winner has .png extension while original was .jpg/.tga:
        if not source_winner.exists():
            png_cand = source_winner.with_suffix(".png")
            if png_cand.exists():
                source_winner = png_cand

        if not source_winner.exists():
            print(f"Warning: winner not found for {uid}: {source_winner}")
            continue

        # Backup original if not already backed up
        bak_file = target_file.with_name(target_file.name + ".orig_bak")
        if target_file.exists() and not bak_file.exists() and not target_file.is_symlink():
            shutil.copy2(target_file, bak_file)

        # Replace target with symlink or copy
        if target_file.is_symlink() or target_file.exists():
            target_file.unlink()

        try:
            os.symlink(source_winner, target_file)
        except OSError:
            shutil.copy2(source_winner, target_file)
        count += 1

    print(f"Installed {count} HD UI textures into {GAME} (with backups).")
    print(f"Skipped {skipped_orig} units marked to use original.")

def uninstall_ui():
    count = 0
    for uid, u in UI_CAT["units"].items():
        rel_src = u["ui_src"]
        target_file = GAME / rel_src
        bak_file = target_file.with_name(target_file.name + ".orig_bak")
        if bak_file.exists():
            if target_file.is_symlink() or target_file.exists():
                target_file.unlink()
            shutil.move(bak_file, target_file)
            count += 1
    print(f"Restored {count} original UI files.")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "install"
    if cmd == "install":
        install_ui()
    elif cmd == "uninstall":
        uninstall_ui()
