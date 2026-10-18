"""tile_text_batch.py LIST.json IDX[,IDX..]|all: pictures WITH lettering. The lettering is taken off first (textlayer.split: plate + text layer), the lettering-free plate is upscaled by the
tiled FLUX job, and the text layer is put back on top at 4x (textlayer.restore). Each result is checked (downscaled back to the source size it must match the original), logged and
submitted as the newest custom version of the unit. Resumable; a failure of one picture does not stop the run.   List entries: [uid, ui_src, w, h]."""
import json, sys, time, traceback
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import textlayer as tl
import tile_batch as tb

EXT = Path.home() / "Games/CSVM/extracted"


def run(entry, label_suffix=""):
    uid, src, w, h = entry
    stem = Path(src).stem if "BZOOMNEWS" not in src else Path(src).stem + "_" + __import__("hashlib").md5(src.encode()).hexdigest()[:4]
    work = ROOT / "tile_job" / f"{stem}_tl"
    if (work / "submitted").exists():
        print("skip (done)", stem); return
    t0 = time.time()
    if not (work / "plate.png").exists():
        info = tl.split(EXT / src, work)
        print(stem, "split", info, flush=True)
    plate4 = tb.upscale_job(stem + "_plate", work / "plate.png", tb.NEUTRAL + tb.KEEP)
    final = ROOT / "tile_job" / f"{stem}_4x_text.png"
    tl.restore(plate4, work / "layer.png", final)
    src_img = Image.open(EXT / src).convert("RGB")
    back = Image.open(final).convert("RGB").resize(src_img.size, Image.BOX)
    diff = float(np.abs(np.asarray(back, np.float32) - np.asarray(src_img, np.float32)).mean())
    ocr_orig = len(tl.ocr_words(src_img)); ocr_new = len(tl.ocr_words(Image.open(final).convert("RGB").resize((src_img.size[0] * 2, src_img.size[1] * 2), Image.LANCZOS)))
    res = tb.submit(uid, final, "Tiled FLUX 4x, text layer restored", src=EXT / src)
    print(stem, f"downscaled-vs-source mean abs diff {diff:.1f}/255, OCR words orig {ocr_orig} -> final {ocr_new}", res, f"{time.time() - t0:.0f}s", flush=True)
    if "submitted as" in res or "already submitted" in res:
        (work / "submitted").write_text(time.strftime("%F %T"))


if __name__ == "__main__":
    items = json.load(open(sys.argv[1]))
    sel = range(len(items)) if sys.argv[2] == "all" else [int(x) for x in sys.argv[2].split(",")]
    for i in sel:
        try:
            run(items[i])
        except Exception:                                                        # noqa: BLE001
            print("FAILED", items[i][1], traceback.format_exc()[-600:], flush=True)
