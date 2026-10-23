"""mag7_figure.py: the photograph of the 'New York Weekly' cover (SB_21_01_MAG7) taken from the shipped redraw, the lettering from the tiled redraw's fixed version.
The tiled 4x redraw drew the woman's windswept stole two and three times over and left ghost outlines (the print is a motion-blurred photo and the redraw kept every
echo as a limb), where the earlier single-pass redraw has one clean stole. Both are straightened (tile_mags_rot frame); the shipped one is registered on the tiled one with
an affine ECC fit, and wins inside the photograph's rectangle except wherever mag_sb_21_01_mag7.py edited lettering (the pixels that differ from the unedited tiled redraw,
dilated and closed into blocks); outside the rectangle (logo, slogan, band) the tiled version stays. Usage: mag7_figure.py FIXED_UPRIGHT.png SHIPPED_UPRIGHT.png OUT_PAGE.png"""
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tile_mags_rot as tr

STEM = "SB_21_01_MAG7"
PHOTO = (300, 420, 1230, 1400)                 # x0, y0, x1, y1 of the photograph in the straightened frame


def feather(m, r):
    return np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), r), 0, 1)


def run(fixed, shipped, dst):
    A = cv2.imread(fixed)
    B = cv2.imread(shipped)
    T = cv2.imread(str(ROOT / "tile_job" / f"rot_{STEM}_4x.png"))
    g = lambda x: cv2.GaussianBlur(cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), (0, 0), 1.2).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    cc, warp = cv2.findTransformECC(g(T), g(B), warp, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), None, 5)
    warp[:, 2] /= 0.25
    Bw = cv2.warpAffine(B, warp, (A.shape[1], A.shape[0]), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    text = (np.abs(A.astype(int) - T.astype(int)).max(2) > 12).astype(np.uint8)
    text = cv2.dilate(text, np.ones((41, 25), np.uint8))
    text = cv2.morphologyEx(text, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
    text[775:855, 300:730] = 1                       # the caption's whole band: the shipped redraw's own caption sits a few px off the fixed one
    box = np.zeros(text.shape, np.float32)
    box[PHOTO[1]:PHOTO[3], PHOTO[0]:PHOTO[2]] = 1
    w = (feather(box, 18) * (1 - feather(text, 6)))[..., None]
    out = (A.astype(np.float32) * (1 - w) + Bw.astype(np.float32) * w).clip(0, 255).astype(np.uint8)
    up = Path(dst).with_name(Path(dst).stem + "_upright.png")
    cv2.imwrite(str(up), out)
    print("ecc", round(float(cc), 3), "page", tr.back(STEM, src=up, dst=dst))


if __name__ == "__main__":
    run(*sys.argv[1:4])
