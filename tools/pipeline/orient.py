"""orient.py IMAGE  -> prints the clockwise rotation (0/90/180/270) that makes any lettering upright, with the evidence.

Deterministic: tesseract is run on the picture at the four rotations (and two enlargements); each rotation scores the
confidence-weighted length of the clean alphabetic words it reads.  A rotation is only chosen when it clearly beats
upright and the other two, otherwise 0 (leave the picture alone).

    from orient import detect_rotation
    rot, info = detect_rotation(pil_image_rgba)      # rot is degrees clockwise to apply to make the text upright
"""
import os, re, subprocess, sys, tempfile
from pathlib import Path
from PIL import Image


ENV = dict(os.environ, TESSDATA_PREFIX=os.path.expanduser("~/.local/share/tessdata"))   # the system folder has no English model


def _read(img, psm):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.png"; img.save(p)
        r = subprocess.run(["tesseract", str(p), "stdout", "--psm", str(psm), "tsv"], capture_output=True, text=True, timeout=40, env=ENV).stdout
    words = []
    for ln in r.splitlines()[1:]:
        f = ln.split("\t")
        if len(f) >= 12:
            w = f[11].strip()
            try:
                c = float(f[10])
            except ValueError:
                continue
            if c >= 70 and len(w) >= 3 and re.fullmatch(r"[A-Za-z][A-Za-z.'’]*", w):
                words.append((w, c))
    return words


def detect_rotation(rgba, strong_conf=80.0, min_strong=2):
    """clockwise degrees that make lettering upright (0 when there is no convincing lettering or it already reads upright)"""
    bg = Image.new("RGBA", rgba.size, (128, 128, 128, 255)); bg.alpha_composite(rgba.convert("RGBA"))
    base = bg.convert("L")
    words = {}
    for rot in (0, 90, 180, 270):
        g = base.rotate(-rot, expand=True)                      # PIL rotates counter-clockwise, so -rot is clockwise
        best = {}
        for k in (6, 10):
            div = 1 if max(g.size) * k <= 1600 else 2
            big = g.resize((max(1, g.width * k // div), max(1, g.height * k // div)), Image.BICUBIC)
            for psm in (11, 6):
                for w, c in _read(big, psm):
                    best[w] = max(best.get(w, 0.0), c)
        words[rot] = best
    strong = {r: sorted(w for w, c in words[r].items() if c >= strong_conf) for r in words}
    score = {r: sum(c * len(w) for w, c in words[r].items()) for r in words}
    cand = [r for r in words if len(strong[r]) >= min_strong]
    rot = 0
    if cand:
        top = max(cand, key=lambda r: (len(strong[r]), score[r]))
        if top != 0 and len(strong[top]) > len(strong[0]):
            rot = top
    return rot, {"strong_words": {k: v for k, v in strong.items() if v}, "chosen": rot}


if __name__ == "__main__":
    rot, info = detect_rotation(Image.open(sys.argv[1]))
    print(rot, info)
