"""tile_hangar.py: the 11 plane-construction hangar frames (PC_P_HANGAR0..10) as one consistent set.
The hangar itself is identical in every frame; only the plane differs. So: a shared background = per-pixel median of the frames (planes cover any pixel in fewer than half of the
frames, so the median is the bare hangar); it is upscaled ONCE. Each frame is upscaled on its own (same tile seeds). The final frame = its own upscale where it differs from the
background (the plane and what it hides) and the SHARED upscaled background everywhere else, with a feathered mask. Every frame therefore has the same hangar, pixel for pixel."""
import json, sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tile_batch as tb

EXT = Path.home() / "Games/CSVM/extracted"
GFX = EXT / "rof/ASSETS/GRAPHICS"


def main(only=None):
    ui = json.load(open(ROOT / "ui_catalog.json"))["units"]
    uid_of = {Path(c["ui_src"]).stem.upper(): u for u, c in ui.items()}
    fr = [Image.open(GFX / f"PC_P_HANGAR{i}.JPG").convert("RGB") for i in range(11)]
    st = np.stack([np.asarray(f).astype(np.float32) for f in fr])
    med = np.median(st, 0)
    work = ROOT / "tile_job" / "hangar"; work.mkdir(exist_ok=True)
    bg_src = work / "bg.png"; Image.fromarray(med.astype(np.uint8)).save(bg_src)
    hint = tb.NEUTRAL + tb.KEEP
    bg4 = np.asarray(Image.open(tb.upscale_job("hangar_bg", bg_src, hint)).convert("RGB"), np.float32)
    H4, W4 = bg4.shape[:2]
    for i in range(11):
        if only is not None and i not in only:
            continue
        if (work / f"submitted_{i}").exists():
            continue
        t0 = time.time()
        fsrc = work / f"frame{i}.png"; fr[i].save(fsrc)
        f4 = np.asarray(Image.open(tb.upscale_job(f"hangar_f{i}", fsrc, hint)).convert("RGB"), np.float32)
        dev = np.abs(st[i] - med).max(-1)
        m = (cv2.GaussianBlur(dev, (0, 0), 1.2) > 26).astype(np.uint8)
        m = cv2.dilate(m, np.ones((9, 9), np.uint8))
        m4 = cv2.resize(m.astype(np.float32), (W4, H4), interpolation=cv2.INTER_CUBIC)
        m4 = np.clip(cv2.GaussianBlur(m4, (0, 0), 7.0) * 1.6, 0, 1)[..., None]
        res = f4 * m4 + bg4 * (1 - m4)
        out = ROOT / "tile_job" / f"PC_P_HANGAR{i}_4x.png"
        Image.fromarray(res.clip(0, 255).astype(np.uint8)).save(out)
        print(f"hangar {i}", tb.submit(uid_of[f"PC_P_HANGAR{i}"], out, "Tiled FLUX 4x, shared hangar background"), f"{time.time() - t0:.0f}s", flush=True)
        (work / f"submitted_{i}").write_text(time.strftime("%F %T"))


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else None)
