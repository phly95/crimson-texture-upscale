"""tile_stamps.py: SB_KILLMARKERCOMBINED (70 x 2200 atlas of 22 rubber-stamp frames, 70 x 100 each) redrawn frame by frame.
Each frame is laid on white paper, redrawn by klein at 7x, then turned back into ink: alpha = how far a pixel is from paper towards the frame's ink colour, colour = that ink colour.
Of the seeds the one whose alpha, shrunk back to 70 x 100, is closest to the original's is kept (OCR cannot read the curved word). 7x because a texture side may not pass 16384 px (2200 x 7 = 15400).
  tile_stamps.py gen [FRAME ...]     redraw frames (default all) into tile_job/killmarkers/
  tile_stamps.py build OUT.png       assemble the atlas from the chosen seeds (chosen.json, else the best-OCR seed)"""
import json, re, subprocess, sys, os, tempfile, difflib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tile_upscale as tu

SRC = Path.home() / "Games/CSVM/extracted/rof/ASSETS/GRAPHICS/SB_KILLMARKERCOMBINED.PNG"
WORK = ROOT / "tile_job" / "killmarkers"
FW, FH, N, K = 70, 100, 22, 7
GW, GH = 512, 720                                               # klein sizes (multiples of 16), resized to 70K x 100K afterwards
NAMES = ["Hoplite", "Hellhound", "Balmoral", "Bloodhawk", "Brigand", "Devastator", "Firebrand", "Fury", "Kestrel", "Peacemaker", "Warhawk"]
PROMPT = ("Restore this rubber-stamp print as a crisp, sharp, high-resolution version of itself on plain white paper. Keep every shape exactly where it is: the ring, the diagonal bar, "
          "the aircraft silhouette inside it, the empty square at the top{star} and the word under the ring{word}. Keep the grungy uneven ink texture of a worn stamp. "
          "One flat ink colour only, white background, no shadows, no extra marks.")


def frames():
    a = np.asarray(Image.open(SRC).convert("RGBA"))
    return [a[i * FH:(i + 1) * FH] for i in range(N)]


def ink_of(f):
    m = f[..., 3] > 200
    return np.median(f[m][:, :3], axis=0) if m.sum() > 20 else np.array([99, 8, 25.])


def on_white(f):
    al = f[..., 3:] / 255.0
    return (f[..., :3] * al + 255 * (1 - al)).clip(0, 255).astype(np.uint8)


def ocr_word(img):
    g = img.convert("L"); w, h = g.size
    crop = g.crop((0, int(h * .70), w, h)); crop = crop.resize((crop.size[0] * 2, crop.size[1] * 2), Image.LANCZOS)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "w.png"; crop.save(p)
        t = subprocess.run(["tesseract", str(p), "stdout", "--psm", "7"], capture_output=True, text=True,
                           env=dict(os.environ, TESSDATA_PREFIX=str(Path.home() / ".local/share/tessdata"))).stdout
    return re.sub(r"[^A-Za-z]", "", t)


def name_of(img):
    w = ocr_word(img)
    sc = [(difflib.SequenceMatcher(None, w.lower(), n.lower()).ratio(), n) for n in NAMES]
    return max(sc)


def to_ink(img, ink):
    """redraw on white -> alpha ramp from paper to ink, flat ink colour"""
    x = np.asarray(img.convert("RGB").resize((FW * K, FH * K), Image.LANCZOS), np.float32)
    paper = np.percentile(x.reshape(-1, 3), 97, axis=0)
    ink = np.asarray(ink, np.float32)
    d = (paper - ink); d[np.abs(d) < 8] = 8
    t = ((paper - x) / d).mean(axis=2)
    a = np.clip((t - 0.04) / 0.92, 0, 1)
    rgb = np.broadcast_to(ink, x.shape)
    return np.dstack([rgb, a * 255]).round().astype(np.uint8)


def gen(idx):
    WORK.mkdir(parents=True, exist_ok=True)
    fs = frames()
    for i in idx:
        f = fs[i]; ink = ink_of(f)
        red = ink[0] > 120 and ink[1] < 60
        ref = WORK / f"f{i:02d}_ref.png"
        Image.fromarray(on_white(f)).resize((GW, GH), Image.LANCZOS).save(ref)
        for s in (1, 2, 3):
            out = WORK / f"f{i:02d}_s{s}.png"
            if out.exists():
                continue
            tu.locked_gen_wh(ref, out, PROMPT.format(star=", the five-pointed star above the ring" if red else "", word=", spelled exactly as in the picture"), GW, GH, 700 + s)
        print("frame", i, "ink", ink.round().tolist(), flush=True)


def score(i, s, fs):
    """mean abs difference between the redraw's alpha shrunk to the original frame size and the original's alpha"""
    a = to_ink(Image.open(WORK / f"f{i:02d}_s{s}.png"), ink_of(fs[i]))[..., 3]
    small = np.asarray(Image.fromarray(a).resize((FW, FH), Image.BOX), np.float32)
    return float(np.abs(small - fs[i][..., 3]).mean())


def build(outp):
    fs = frames(); chosen = json.loads((WORK / "chosen.json").read_text()) if (WORK / "chosen.json").exists() else {}
    atlas = np.zeros((FH * K * N, FW * K, 4), np.uint8)
    for i in range(N):
        best = None
        for s in (1, 2, 3):
            p = WORK / f"f{i:02d}_s{s}.png"
            if p.exists():
                sc = score(i, s, fs)
                if best is None or sc < best[0]:
                    best = (sc, s)
        s = chosen.get(str(i), best[1] if best else None)
        if s is None:
            a = Image.fromarray(fs[i]).resize((FW * K, FH * K), Image.LANCZOS)
            atlas[i * FH * K:(i + 1) * FH * K] = np.asarray(a)
        else:
            atlas[i * FH * K:(i + 1) * FH * K] = to_ink(Image.open(WORK / f"f{i:02d}_s{s}.png"), ink_of(fs[i]))
        print("frame", i, "seed", s, "diff", None if best is None else round(best[0], 1))
    Image.fromarray(atlas).save(outp)


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen([int(x) for x in sys.argv[2:]] or range(N))
    else:
        build(sys.argv[2])
