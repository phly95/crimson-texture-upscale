"""typeset_report.py IN.png OUT.png: SB_24_01_PREPORT (the Empire State police report). The thumbnail's typed fields and body are garbled and its baked text is not the game's wording:
the game string is cut off in the middle of its last sentence and the thumbnail runs the page off the bottom. The header fields and the body are re-set from the game string in
Courier New (bold labels), wiping only the dark ink of the old text and keeping the form's rules, the signature, the stamps and the torn edge. Two form labels with no string
('Appoiid-920', 'Arr.202') are written as plausible form furniture."""
import re, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import typeset_page as tp

COUR, COURB = str(tp.F / "cour.ttf"), str(tp.F / "courbd.ttf")
SIZE = 23
ADV = 0.6 * SIZE
LEFT, TOP, PITCH = 68, 455, 29.7
CAP = 63

FIELDS = [(("CASE No:", " 234899323"), 68, 277), (("OFFICER of RECORD:", " Det. Seiden"), 68, 336), (("Page:", " 3 of 3"), 68, 395)]
OLD = [(60, 262, 335, 294), (60, 322, 505, 352), (60, 382, 232, 410), (78, 95, 242, 119), (28, 190, 125, 218), (45, 432, 965, 928)]
PRE = "Form 920-DCI"


def body_lines():
    t = tp.STR[40110].split("<B>SUMMARY (cont):<b>", 1)[1]
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n(?=\s{5})|\n\n", t) if p.strip()]
    out = []                                                           # (text, bold_prefix_len)
    for i, p in enumerate(paras):
        p = p.replace("<B>", "").replace("<b>", "")
        if i == 0:
            p = "SUMMARY (cont):  " + p
        if p.startswith("CONCLUSIONS:"):
            out.append(("", 0))
        ls = tp.wrap(p, CAP)
        for k, ln in enumerate(ls):
            out.append((ln, 1 if (k == 0 and (i == 0 or p.startswith("CONCLUSIONS:"))) else 0))
    return out


def run(src, dst):
    im = np.asarray(Image.open(src).convert("RGBA"))
    H, W = im.shape[:2]
    rgb = im[..., :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = (gray < 150) & (im[..., 3] > 250)
    hr = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (70, 1))) > 0
    vr = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))) > 0
    rules = cv2.dilate((hr | vr).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    wipe = np.zeros((H, W), bool)
    for (x0, y0, x1, y1) in OLD:
        wipe[y0:y1, x0:x1] = (dark & ~rules)[y0:y1, x0:x1]
    ink = np.median(rgb[dark & ~rules][:], axis=0) * 0.7
    f, fb = ImageFont.truetype(COUR, SIZE * 4), ImageFont.truetype(COURB, SIZE * 4)
    cv_ = Image.new("L", (W * 4, H * 4), 0); dr = ImageDraw.Draw(cv_)
    for (lab, val), x, y in FIELDS:
        dr.text((x * 4, y * 4), lab, font=fb, fill=240, anchor="lm")
        dr.text((x * 4 + fb.getlength(lab), y * 4), val, font=f, fill=240, anchor="lm")
    dr.text((80 * 4, 106 * 4), PRE, font=ImageFont.truetype(COUR, 19 * 4), fill=235, anchor="lm")
    dr.text((34 * 4, 205 * 4), "Date:", font=ImageFont.truetype(COUR, 19 * 4), fill=235, anchor="lm")
    y = TOP
    for ln, bold in body_lines():
        if ln:
            if bold:
                lab = "SUMMARY (cont):" if ln.startswith("SUMMARY") else "CONCLUSIONS:"
                dr.text((LEFT * 4, y * 4), lab, font=fb, fill=240, anchor="lm")
                dr.text((LEFT * 4 + fb.getlength(lab), y * 4), ln[len(lab):], font=f, fill=240, anchor="lm")
            else:
                dr.text((LEFT * 4, y * 4), ln, font=f, fill=240, anchor="lm")
        y += PITCH
    la = np.asarray(cv_.resize((W, H), Image.LANCZOS), np.float32) / 255
    la = np.clip(cv2.GaussianBlur(la, (0, 0), 0.45) * 1.3, 0, 1)
    old = cv2.dilate(wipe.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    keep = (~old & (gray > 140) & (im[..., 3] > 250)).astype(np.float32)
    pg = rgb.astype(np.float32)
    smooth = cv2.GaussianBlur(pg * keep[..., None], (0, 0), 3) / np.maximum(cv2.GaussianBlur(keep, (0, 0), 3), 1e-3)[..., None]
    noise = cv2.GaussianBlur(np.random.default_rng(3).normal(0, 1.0, (H, W)).astype(np.float32), (0, 0), 0.8)[..., None] * 2.0
    base = np.where(old[..., None], smooth + noise, pg)
    out = base * (1 - la[..., None]) + ink[None, None, :] * la[..., None]
    res = im.copy(); res[..., :3] = out.clip(0, 255).astype(np.uint8)
    Image.fromarray(res).save(dst)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
