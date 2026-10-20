"""headtex_gen.py STEM SEED [PROMPT_TAIL]: layout-locked klein redraw of a character UV texture (original = reference, tile_upscale-style prompt) -> face_audit/headtex/<stem>_s<seed>.png"""
import sys, json
from pathlib import Path
import cv2, numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "tools"))
import tile_upscale as T
cat = json.load(open(ROOT / "catalog.json"))["units"]
def uid_of(stem): return next(u for u, v in cat.items() if v["stem"] == stem and not v.get("is_mip"))
def ref_of(stem, S=1024):
    o = Image.open(ROOT / "pipeline/units" / uid_of(stem) / "orig.png").convert("RGB")
    k = S / max(o.size); return o.resize((round(o.width * k), round(o.height * k)), Image.BICUBIC)
if __name__ == "__main__":
    stem, seed = sys.argv[1], int(sys.argv[2]); tail = sys.argv[3] if len(sys.argv) > 3 else ""
    ref = ref_of(stem); D = ROOT / "face_audit/headtex"; rp = D / f"{stem}_ref.png"; ref.save(rp)
    prompt = T.BASE_PROMPT + " This is a flattened UV texture sheet for a 3D model, not a portrait: do not re-frame, re-scale, move or re-pose anything, and do not add new objects. " + tail
    out = D / f"{stem}_s{seed}.png"
    T.locked_gen_wh(str(rp), str(out), prompt, ref.width // 16 * 16, ref.height // 16 * 16, seed)
    print(out)
