"""tile_mags_rot.py [--prep|--run|--back] [STEM ...]: the magazine covers straightened before the tiled FLUX 4x, rotated back after.
Klein draws upright lettering far better than lettering on a tilt, so each cover is deskewed (ANGLE, cv2 degrees, positive = counter-clockwise; fitted on the paper edge and checked
against the printed lines) on an expanded canvas, redrawn by tile_batch.upscale_job as tile_job/rot_<STEM>, and the result is rotated back with the exact inverse of the 4x-scaled
transform and cut to the original's own alpha. --prep writes the straightened sources and a contact sheet, --run does the jobs (no quoted lettering in the hint), --back writes
tile_job/rot_<STEM>_page.png (RGBA, 4x the original's size). back(stem, src, dst) takes an edited copy of the straightened picture."""
import sys, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SB = Path.home() / "Games/CSVM/extracted/rof/ASSETS/GRAPHICS/SCRAPBOOK"
ANGLE = {"SB_00_00_MAG1": -7.2, "SB_01_02_MAG2": 6.2, "SB_11_01_MAG3": -4.9, "SB_11_01_MAG4": 5.9, "SB_14_01_MAG5": -4.0, "SB_21_01_MAG7": -8.4, "SB_19_01_ZPJCOVER": 4.2, "SB_22_01_COMICOVER": 7.2}
PAD = 8
PLAIN = ("This is a detail crop of a larger picture. Redraw only what is visible in this crop; never add anything that is not already in the crop. It shows a vintage printed magazine "
         "cover with worn paper edges. Keep every printed letter exactly as it appears in the crop; never write a word or number that is not already in the crop, and lettering the crop "
         "cuts off stays cut off. Tiny body text stays tiny soft grey lines. Do not add any lettering, logos or creases that are not already in the crop, and do not change anyone's "
         "face, pose or clothing.")


def forward(W, H, ang):
    """affine of the original (W x H) into the expanded straightened canvas, and the canvas size"""
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    pts = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], np.float64) @ M.T
    x0, y0 = pts.min(0) - PAD
    x1, y1 = pts.max(0) + PAD
    M[:, 2] -= (x0, y0)
    return M, (int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0)))


def extend(rgba):
    """the colour under a cover's transparent part is whatever the scene behind it was (desk, papers); replace it with a smooth extension of the paper's own edge colour"""
    rgb = rgba[..., :3].astype(np.float32)
    keep = (cv2.erode((rgba[..., 3] > 128).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0).astype(np.float32)
    out, filled = rgb * keep[..., None], keep.copy()
    for sg in (3, 8, 24, 80):
        num = cv2.GaussianBlur(rgb * keep[..., None], (0, 0), sg)
        den = cv2.GaussianBlur(keep, (0, 0), sg)
        new = (filled < 0.5) & (den > 0.02)
        out[new] = (num / np.maximum(den, 1e-6)[..., None])[new]
        filled[new] = 1
    return np.where(filled[..., None] > 0.5, out, rgb).clip(0, 255).astype(np.uint8)


def prep(stem):
    im = Image.open(SB / (stem + ".PNG")).convert("RGBA")
    W, H = im.size
    M, (W2, H2) = forward(W, H, ANGLE[stem])
    rgb = cv2.warpAffine(extend(np.asarray(im)), M, (W2, H2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    out = ROOT / "tile_job" / f"rot_{stem}_src.png"
    Image.fromarray(rgb).save(out)
    return out


def run(stem):
    import tile_batch as tb
    t0 = time.time()
    out = tb.upscale_job("rot_" + stem, prep(stem), PLAIN)
    print(stem, out, f"{time.time() - t0:.0f}s", flush=True)


def back(stem, src=None, dst=None):
    """src: the straightened 4x picture to rotate back (default tile_job/rot_<STEM>_4x.png, or an edited copy of it), dst: the page-frame output"""
    im = Image.open(SB / (stem + ".PNG")).convert("RGBA")
    W, H = im.size
    M, (W2, H2) = forward(W, H, ANGLE[stem])
    big = np.asarray(Image.open(src or ROOT / "tile_job" / f"rot_{stem}_4x.png").convert("RGB"))
    S = np.diag([4.0, 4.0, 1.0])
    M4 = (S @ np.vstack([M, [0, 0, 1]]) @ np.linalg.inv(S))[:2]
    res = cv2.warpAffine(big, M4, (W * 4, H * 4), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    a = im.getchannel("A").resize((W * 4, H * 4), Image.LANCZOS)
    out = Path(dst) if dst else ROOT / "tile_job" / f"rot_{stem}_page.png"
    Image.merge("RGBA", (*Image.fromarray(res).split(), a)).save(out)
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    stems = sys.argv[2:] or list(ANGLE)
    for s in stems:
        {"--prep": prep, "--run": run, "--back": back}[mode](s)
    if mode == "--prep":
        ims = [Image.open(ROOT / "tile_job" / f"rot_{s}_src.png") for s in stems]
        sheet = Image.new("RGB", (4 * 420, 2 * 520), (40, 40, 40))
        for i, m in enumerate(ims):
            k = min(410 / m.width, 510 / m.height)
            sheet.paste(m.resize((int(m.width * k), int(m.height * k)), Image.LANCZOS), ((i % 4) * 420, (i // 4) * 520))
        sheet.save(ROOT / "tile_job" / "rot_prep_sheet.png")
