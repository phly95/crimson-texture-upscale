"""headtex_cpmerge.py TARGET [sl sab]: drawn bake (headtex/<tg>_draw_bake.png) over the filtered texture (<tg>_toonf.png): drawn texels' broad Lab tone pulled to the base's, feathered edge -> headtex/<tg>_drawn.png"""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_fin as F
T = F.T; D = F.H.ROOT / "face_audit/headtex"
def merge(tg, sl=12.0, sab=30.0, feather=2.5):
    bk = np.array(Image.open(D / f"{tg}_draw_bake.png")); base = np.array(Image.open(D / f"{tg}_toonf.png").convert("RGB"))
    a = (bk[..., 3] > 127).astype(np.uint8); a = cv2.erode(a, np.ones((3, 3), np.uint8)); a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), feather)[..., None]
    fill = cv2.inpaint(bk[..., :3], (bk[..., 3] < 127).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)                       # keep colour-lock stats sane over the holes
    drawn = np.clip(T.colour_lock(fill / 255., base / 255., sl, sab), 0, 1) * 255
    return np.clip(drawn * a + base * (1 - a) + .5, 0, 255).astype(np.uint8)
if __name__ == "__main__":
    tg = sys.argv[1]; Image.fromarray(merge(tg, *[float(x) for x in sys.argv[2:4]])).save(D / f"{tg}_drawn.png")
