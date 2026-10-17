"""textcheck.py UNIT_DIR [seed]: decide whether a redraw needs a review.  Cheap, high-recall triggers (the VLM review is the real judge):
  - lettering: the prompt asks for words (quoted or capitalised) -> every such redraw is looked at once
  - stray text: OCR finds digits or confident words in the redraw that the prompt never asked for (e.g. coordinates printed on the picture)
  - OCR could not find a requested word (misspelling shows up here)"""
import difflib, os, re, subprocess, sys, tempfile
from pathlib import Path
from PIL import Image

STOP = {"RGB", "PNG", "HD", "UV", "RGBA", "II", "III", "NOT", "THE", "AND", "ONE", "ALL", "NO", "AN", "OF", "UK", "USA", "US", "RAF", "ID", "OK", "TV"}


def expected_words(prompt):
    p = prompt
    out = []
    for m in re.finditer(r'[“"]([^”"\n]{2,40})[”"]', p):                    # quoted lettering
        out += re.findall(r"[A-Za-z][A-Za-z'.]{2,}", m.group(1))
    for m in re.finditer(r"\b(?:[A-Za-z]-){2,}[A-Za-z]\b", p):               # letter-by-letter spellings
        out.append(m.group(0).replace("-", ""))
    out += [w for w in re.findall(r"\b[A-Z][A-Z'.]{2,}\b", p) if w not in STOP]
    seen, res = set(), []
    for w in out:
        k = re.sub(r"[^A-Z]", "", w.upper())
        if len(k) >= 3 and k not in seen and k not in STOP:
            seen.add(k); res.append(k)
    return res


def ocr_tokens(img, min_conf=0):
    env = dict(os.environ, TESSDATA_PREFIX=str(Path.home() / ".local/share/tessdata"))
    toks = []
    g = img.convert("L")
    for inv in (False, True):
        im2 = Image.eval(g, lambda v: 255 - v) if inv else g
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.png"; im2.save(p)
            r = subprocess.run(["tesseract", str(p), "stdout", "--psm", "11", "tsv"], capture_output=True, text=True, timeout=60, env=env).stdout
        for ln in r.splitlines()[1:]:
            f = ln.split("\t")
            if len(f) >= 12 and f[11].strip():
                try:
                    c = float(f[10])
                except ValueError:
                    continue
                if c >= min_conf:
                    toks.append((f[11].strip(), c))
    return toks


def flatten(img_path):
    im = Image.open(img_path).convert("RGB")
    return im


def check(d, seed, prompt):
    d = Path(d)
    exp = expected_words(prompt)
    im = flatten(d / f"klein_s{seed}.png")
    toks = ocr_tokens(im)
    letters = [(re.sub(r"[^A-Z]", "", t.upper()), c, t) for t, c in toks]
    missing = []
    for e in exp:
        best = max((difflib.SequenceMatcher(None, e, l).ratio() for l, c, t in letters if l and c >= 30), default=0.0)
        if best < 0.75:
            missing.append(e)
    stray = []
    for t, c in toks:
        if c < 70:
            continue
        L = re.sub(r"[^A-Z]", "", t.upper())
        if re.search(r"\d", t) and len(t) >= 2:
            stray.append(t)                                                   # digits: coordinates, percentages, RGB values leaking into the picture
        elif len(L) >= 3 and c >= 85 and not any(difflib.SequenceMatcher(None, e, L).ratio() >= 0.75 for e in exp):
            stray.append(t)
    return {"expected": exp, "missing": missing, "stray": sorted(set(stray)), "ocr": [t for t, c in toks if c >= 70][:20],
            "trigger": bool(exp or stray)}


if __name__ == "__main__":
    d = Path(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    prompt = re.search(r"PROMPT:\s*(.+)", (d / "prompt.txt").read_text(), re.S).group(1)
    print(check(d, seed, prompt))
