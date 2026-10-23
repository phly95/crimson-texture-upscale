"""parts_comi.py: COMICOVER (Nathan Zachary: Fortune Hunter!) rebuilt from parts -> pipeline/mag_parts/COMICOVER/
 The cover is flat comic art: a diffusion redraw only adds glow and invented ladder/boot detail, so the art part is the ORIGINAL's own pixels (deskewed, edge-preserving smooth, unsharp + edge
 steepening, 4x), and the small lettering (slogan, blurb: unreadable in the print) is wiped and retyped from the game strings by the existing mag_sb_22 routines.
 parts: page (earlier redraw, hidden under) | art (original, whole cover) | slogan+blurb are baked into art's text step and also kept as a separate text part"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mag_parts as mp
import mag_sb_22_01_comicover as ms

C = "COMICOVER"
D = mp.PARTS / C


def art_upright():
    u = mp.upright_1x(C)
    return np.asarray(mp.flat_up(u, 4, sigma_s=3.0, sigma_r=0.08, amount=0.9, radius=1.5, steep=0.5))


def slogan(img):
    """the three slogan lines: ALL of the old ink incl. its soft grey fringe (anything clearly darker than the paper, grown 4 px) is replaced by the paper's own colour (normalised average of clean paper), orange title outline kept, then the lines are set"""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32); h, s_, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    box = np.zeros((H, W), bool); box[316:432, 420:640] = True
    paper = (v > 205) & (s_ < 110) & box
    orange = cv2.dilate((((h >= 2) & (h <= 18) & (s_ >= 110) & (v >= 110)).astype(np.uint8)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    outline = cv2.dilate((v < 90).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    old = box & ~paper & ~orange
    old = cv2.dilate(old.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    old &= box & ~orange
    keep = paper & ~old
    out = ms.paper_fill(img, old, keep, sigma=6.0)
    alpha = np.zeros((H, W), np.float32)
    for text, xr, base in ms.SLOGAN:
        alpha = np.maximum(alpha, ms.line_alpha((H, W), text, ms.SLOGAN_PX, base, x_right=xr))
    return ms.paint_colour(out, alpha, ms.SLOGAN_INK, soft=0.6)


def blurb(img):
    """the blurb of ms.fix_blurb, with a plain wipe: the print's ink inside the box (dark or orange, not the statue's green) is replaced by the paper's own colour, then the lines are set"""
    from PIL import ImageFont
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32); h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    box = np.zeros((H, W), bool); box[490:612, 1100:1480] = True
    green = (h >= 35) & (h <= 100) & (s >= 50)
    paper = (s < 95) & (v > 170) & box
    pc = np.median(img[paper], axis=0)
    ink = box & ~green & ((v < 190) | (s >= 95) | (h <= 20) & (s >= 70))
    ink = cv2.dilate(ink.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    ink &= ~green
    out = img.copy().astype(np.float32)
    m = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), 1.2)[..., None]
    out = out * (1 - m) + pc[None, None] * m
    out = out.clip(0, 255).astype(np.uint8)
    f = ImageFont.truetype(str(ms.FONT), int(round(ms.BLURB_PX * 4)))
    l2, _, r2, _ = f.getbbox(ms.BLURB[1][0][0][0], anchor="ls")
    sx = ms.BLURB_L2_WIDTH / ((r2 - l2) / 4)
    for segs, base in ms.BLURB:
        x = ms.BLURB_X - sx * f.getbbox(segs[0][0], anchor="ls")[0] / 4
        for seg, colour in segs:
            a = ms.line_alpha((H, W), seg, ms.BLURB_PX, base, x_origin=x, sx=sx)
            out = ms.paint_colour(out, a, colour)
            x += sx * f.getlength(seg) / 4
    return out


def build():
    art = art_upright()
    clean = art.copy()
    txt = blurb(slogan(art))
    rgba = Image.fromarray(np.dstack([txt, np.full(txt.shape[:2], 255, np.uint8)]), "RGBA")
    pg, x, y = mp.to_page(rgba, C, 0, 0, margin=0)
    mf = mp.Manifest(C)
    mf.add("art", "Cover art, title, badge and lettering (original pixels)", "figure", pg, x, y, 10,
           {"type": "orig-crop", "method": "the original's own pixels, deskewed: bilateral smooth + unsharp + edge steepening at 4x (flat comic art needs no redraw; Klein only added glow and invented ladder/boot detail); slogan and blurb wiped and retyped from the game strings (Arial Bold on the printed baselines)",
            "tool": "tools/parts_comi.py (mag_parts.flat_up, mag_sb_22_01_comicover.fix_slogan/fix_blurb)"}, "problem")


def badge_text(im, x, y):
    """the band's tiny 'NATHAN ZACHARY' and the '3 cents' are blurred smudges in a 361 px print: the dark ink in the band and the price box is inpainted out and both are retyped (game wording)"""
    from PIL import ImageFont, ImageDraw
    a = np.asarray(im).copy(); rgb = a[..., :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    dark = (hsv[..., 2] < 135).astype(np.uint8)
    pm = np.zeros(dark.shape, np.uint8)
    cv2.fillPoly(pm, [np.array([(418 - x, 106), (428 - x, 140), (528 - x, 10), (548 - x, 24), (546 - x, 44), (440 - x, 140), (420 - x, 136)], np.int32) * np.array([1, 1]) * 1 if False else
                      np.array([(414 - x, 118 - y), (426 - x, 140 - y), (534 - x, 10 - y), (548 - x, 22 - y), (548 - x, 46 - y), (440 - x, 142 - y), (416 - x, 136 - y)], np.int32)], 1)
    cv2.fillPoly(pm, [np.array([(452 - x, 100 - y), (526 - x, 100 - y), (526 - x, 156 - y), (452 - x, 156 - y)], np.int32)], 1)
    m = cv2.dilate(dark & pm, np.ones((13, 13), np.uint8)) & pm
    rgb = cv2.inpaint(np.ascontiguousarray(rgb), m, 9, cv2.INPAINT_NS)
    # the price box is flat orange: replace its whole patch by the orange's median so no ghost ring of the old numerals remains
    box = np.zeros(dark.shape, np.uint8); cv2.rectangle(box, (456 - x, 100 - y), (524 - x, 154 - y), 1, -1)
    og = (cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 0] <= 20) & (cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1] > 120) & (box > 0)
    if og.any():
        med = np.median(rgb[og], axis=0); sm = cv2.GaussianBlur(((box > 0) & (m > 0)).astype(np.float32), (0, 0), 3)[..., None]
        rgb = (rgb * (1 - sm) + med[None, None] * sm).astype(np.uint8)
    # band fill (cream) where the band was: the inpaint of thin dark letters on cream keeps cream; text laid on top
    out = Image.fromarray(np.dstack([rgb, a[..., 3]]), "RGBA")
    lay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    f = ImageFont.truetype(str(mp.WIN / "georgiab.ttf") if (mp.WIN / "georgiab.ttf").exists() else str(mp.WIN / "TIMESBD.TTF"), 150)
    t = Image.new("L", (600, 220), 0); ImageDraw.Draw(t).text((6, 200), "3\u00a2", font=f, fill=255, anchor="ls")
    bb = t.getbbox(); t = t.crop(bb); tw = 46; t = t.resize((tw * 4 // 4 * 1, int(t.height * tw / t.width)), Image.LANCZOS)
    col = Image.new("RGBA", t.size, (44, 38, 40, 0)); col.putalpha(t)
    lay.alpha_composite(col, (int(466 - x), int(118 - y)))
    # band text along the band centre line (422,124) -> (540,26)
    import math
    ang = math.degrees(math.atan2(26 - 124, 540 - 422)); L = math.hypot(118, 98)
    f2 = ImageFont.truetype(str(mp.WIN / "arialbd.ttf"), 22)
    txt = "NATHAN ZACHARY"; w = f2.getlength(txt); sc = (L - 14) / w
    t2 = Image.new("L", (int(w) + 8, 30), 0); ImageDraw.Draw(t2).text((4, 24), txt, font=f2, fill=255, anchor="ls")
    t2 = t2.resize((int((w + 8) * sc), int(30 * sc * 1.0)), Image.LANCZOS).rotate(-ang, expand=True, resample=Image.BICUBIC)
    c2 = Image.new("RGBA", t2.size, (36, 32, 30, 0)); c2.putalpha(t2.point(lambda v: int(v * 0.92)))
    cx, cy = (422 + 540) / 2 - x + 0, (26 + 124) / 2 - y
    lay.alpha_composite(c2, (int(cx - t2.width / 2 + 2), int(cy - t2.height / 2 - 2)))
    out.alpha_composite(lay)
    return out


def corner():
    """the price badge (diagonal band, 3 cents) at the torn top-left corner straight from the original's page frame (no deskew round trip, which washed the orange out at the torn edge)"""
    mf = mp.Manifest(C)
    box = (100, 0, 142, 42)
    im, x, y = mp.box_part(C, box, feather=2, amount=0.9, radius=1.5, steep=0.5, rgb=mp.orig_ext_rgb(C))
    im = mp.alpha_clip(im, C, x, y, erode=1)
    im = badge_text(im, x, y)
    mf.add("badge", "Price badge with diagonal band", "emblem", im, x, y, 30, {"type": "orig-crop", "method": "the original's own pixels in the page frame: Lanczos 4x, unsharp, edge steepening, clipped to the torn paper edge", "tool": "mag_parts.box_part", "orig_box": list(box)}, "problem")


if __name__ == "__main__":
    build()
    corner()
    mp.compose(C, str(D / "composed.png"))
