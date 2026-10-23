"""mag4_edge.py IN_PAGE.png OUT_PAGE.png: the right edge of the Confidential cover (SB_11_01_MAG4) put back from the original.
The deskewed redraw lost the three things that sit at that torn edge: the white curled flap over the end of the logo, and the card fan, star and skull below it (they
came back as blurs). The flap (bright, unsaturated pixels of the original, top right) is taken from the original at 4x, and the card/star/skull box from the first,
unrotated tiled redraw (tile_job/mag_SB_11_01_MAG4_4x.png), which drew them correctly; both are feathered into the lettering-fixed page, so its text is kept."""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = Path.home() / "Games/CSVM/extracted/rof/ASSETS/GRAPHICS/SCRAPBOOK/SB_11_01_MAG4.PNG"


def feather(mask, r):
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), r), 0, 1)


def run(src, dst):
    page = Image.open(src).convert("RGBA")
    W, H = page.size
    pg = np.asarray(page).astype(np.float32)
    orig = Image.open(SRC).convert("RGBA")
    ou = np.asarray(orig.resize((W, H), Image.LANCZOS)).astype(np.float32)
    first = np.asarray(Image.open(ROOT / "tile_job/mag_SB_11_01_MAG4_4x.png").convert("RGB")).astype(np.float32)
    out = pg.copy()
    # card fan, star, skull: a box down the right edge, from the first redraw
    box = np.zeros((H, W), np.float32)
    box[int(H * 0.30):int(H * 0.60), int(W * 0.74):] = 1
    m = feather(box, 10)[..., None]
    out[..., :3] = out[..., :3] * (1 - m) + first * m
    # the flap: bright, unsaturated paper of the original in the top right, largest blob
    rgb = ou[..., :3]
    L = rgb.mean(2)
    sat = rgb.max(2) - rgb.min(2)
    reg = np.zeros((H, W), bool)
    reg[int(H * 0.04):int(H * 0.36), int(W * 0.78):] = True
    cand = (L > 165) & (sat < 60) & reg & (ou[..., 3] > 128)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8))
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    flap = cv2.dilate((lab == big).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
    soft = cv2.GaussianBlur(ou[..., :3], (0, 0), 0.8)
    sharp = np.clip(ou[..., :3] + 0.7 * (ou[..., :3] - soft), 0, 255)
    m = feather(flap, 2.0)[..., None]
    out[..., :3] = out[..., :3] * (1 - m) + sharp * m
    out[..., 3] = pg[..., 3]
    Image.fromarray(out.clip(0, 255).astype(np.uint8), "RGBA").save(dst)
    print("flap px", int(flap.sum()))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
