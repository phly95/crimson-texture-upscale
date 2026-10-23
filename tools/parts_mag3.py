"""parts_mag3.py: MAG3 (Hollywood Confidential, purple, plane) rebuilt from parts -> pipeline/mag_parts/MAG3/  (python tools/parts_mag3.py [portrait|plane|icons|all] [force])
 parts: page (approved redraw, base) | portrait | plane | headline (the gold cover lines lifted from the page so they stay over the plane) | cards, star, skull (shared vector icons)"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mag_parts as mp

C = "MAG3"
STEM = "SB_11_01_MAG3"
D = mp.PARTS / C
W = D / "work"
PLANE_BOX = (96, 104, 240, 248)                   # original px, 144x144
PORT_BOX = (10, 66, 58, 114)                      # original px, 48x48
PORT_PROMPT = (mp.tu.BASE_PROMPT + " This is a small vintage sepia head-and-shoulders portrait photograph of a man facing the camera in a leather flying cap with goggles pushed up on his forehead, a long white scarf "
               "hanging down both sides of his chest over a dark jacket. No hands are visible: his arms are down, out of the picture.")
PORT_SEED = 62


def rgb_orig():
    return mp.opaque_on_grey(mp.SCRAPBOOK / (STEM + ".PNG")).copy()


def plane_source():
    """the plane region of the ORIGINAL with the gold cover-line letters, the body-text fragments at the left and the icons (bottom right) inpainted out; the crease is kept"""
    rgb = rgb_orig(); f = rgb.astype(np.int16)
    gold = ((f[..., 0] > 140) & (f[..., 1] > 100) & (f[..., 2] < 100) & (f[..., 0] - f[..., 2] > 70)).astype(np.uint8); gold[170:] = 0
    gold = cv2.dilate(gold, np.ones((5, 5), np.uint8))
    text = mp.strokes_mask(rgb, poly=[(90, 150), (112, 150), (112, 250), (90, 250)], k=5, th=28, grow=2)
    icons = np.zeros(gold.shape, np.uint8); cv2.fillPoly(icons, [np.array([(199, 204), (250, 204), (250, 252), (199, 252)], np.int32)], 1)
    m = np.maximum.reduce([gold, text, icons])
    return Image.fromarray(mp.inpaint(rgb, m, 5)).crop(PLANE_BOX)


def plane(force=False):
    """the plane from the original's own pixels (it is a motion-blurred photograph: any redraw invents a different aeroplane): non-local-means denoise at original size, Lanczos 4x, light unsharp,
    fine grain like the page's; matte = the plane and a band of dark backdrop round it, kept clear of the icons and of the text column"""
    W.mkdir(parents=True, exist_ok=True)
    src = plane_source(); src.save(W / "plane_src.png")
    dn = cv2.fastNlMeansDenoisingColored(np.asarray(src), None, 4, 4, 5, 15)
    big = np.asarray(Image.fromarray(dn).resize((src.width * 4, src.height * 4), Image.LANCZOS)).astype(np.float32)
    big = big + 0.8 * (big - cv2.GaussianBlur(big, (0, 0), 2.2))
    big += np.random.default_rng(3).normal(0, 1.8, big.shape[:2])[..., None]
    big = np.clip(big, 0, 255)
    g = cv2.GaussianBlur(cv2.cvtColor(big.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 3)
    pl = (g > 38).astype(np.uint8)
    pl = cv2.morphologyEx(pl, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(pl)
    keep = np.zeros_like(pl)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > 4000:
            keep |= (lab == i).astype(np.uint8)
    ring = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))
    ox, oy = PLANE_BOX[0] * 4, PLANE_BOX[1] * 4
    cut = np.ones_like(ring)
    cv2.fillPoly(cut, [np.array([(780 - ox, 780 - oy), (big.shape[1], 780 - oy), (big.shape[1], big.shape[0]), (780 - ox, big.shape[0])], np.int32)], 0)   # the icons' corner
    cv2.fillPoly(cut, [np.array([(0, 0), (436 - ox, 0), (436 - ox, big.shape[0]), (0, big.shape[0])], np.int32)], 0)                                        # the text column
    a = cv2.GaussianBlur((ring * cut).astype(np.float32), (0, 0), 14)
    rgba = Image.fromarray(np.dstack([big.astype(np.uint8), (a * 255).astype(np.uint8)]), "RGBA")
    mf = mp.Manifest(C)
    mf.add("plane", "Motion-blurred aeroplane photo", "photo", rgba, ox, oy, 10,
           {"type": "orig-crop", "method": "the original's own pixels (gold lettering, text fragments and icons inpainted out): NL-means denoise at original size, Lanczos 4x, light unsharp, grain. A redraw was tried and rejected: it turned the blur into a different, invented biplane",
            "tool": "tools/parts_mag3.py plane", "orig_box": list(PLANE_BOX)}, "problem")


HEAD_LINES = ["Air Pirate", "Nathan Zachary", "Wows Audiences!"]         # game string 40140
HEAD_ANGLE = -20.0                                                        # the lines rise to the right at 20 degrees (measured on the original's gold strokes)


def headline_geometry():
    """per line: (x0, baseline, x1) of the original's gold strokes on the baseline-levelled axis, mapped back to 4x page coordinates; cap height from the stroke band"""
    rgb = rgb_orig().astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gold = ((np.clip((r - b - 40) / 90, 0, 1) * np.clip((g - 60) / 60, 0, 1)) > 0.5).astype(np.uint8); gold[:55] = 0; gold[142:] = 0; gold[:, :80] = 0; gold[:, 228:] = 0; gold[:75, 175:] = 0       # (not the date, top right)
    c = (150, 100); M = cv2.getRotationMatrix2D(c, HEAD_ANGLE, 1); rot = cv2.warpAffine(gold.astype(np.float32), M, (300, 200)) > 0.3
    Mi = cv2.invertAffineTransform(M)
    prof = rot.sum(1); rows = np.nonzero(prof > 0)[0]
    bands, start = [], rows[0]
    for a_, b_ in zip(rows[:-1], rows[1:]):
        if b_ - a_ > 3:
            bands.append((start, a_)); start = b_
    bands.append((start, rows[-1]))
    bands = [bd for bd in bands if prof[bd[0]:bd[1] + 1].sum() > 200]
    out = []
    for y0, y1 in bands[:3]:
        sub = rot[y0:y1 + 1]; xs = np.nonzero(sub.any(0))[0]
        base = y0 + np.percentile(np.nonzero(sub.sum(1) > sub.sum(1).max() * 0.35)[0], 100) + 0.5                  # last strongly inked row = baseline
        p0 = Mi @ np.array([xs.min(), base, 1.0]); p1 = Mi @ np.array([xs.max() + 1, base, 1.0])
        out.append((p0[0] * 4, p0[1] * 4, p1[0] * 4, p1[1] * 4))
    return out


def headline():
    """the gold cover lines retyped: Times, gold, along the baselines measured on the original; replaces the redraw's lettering, whose dim ghost copies cross the plane"""
    mf = mp.Manifest(C)
    geo = headline_geometry()
    from PIL import ImageFont
    lay_all = None
    top = None
    layers = []
    for text, (x0, y0, x1, y1) in zip(HEAD_LINES, geo):
        slope = (y1 - y0) / (x1 - x0)
        lay, ox, oy = mp.text_layer(tuple(mf.d["size"]), text, x0, y0, x1, slope, 44, mp.WIN / "times.ttf", (232, 178, 40), soft=0.6, space=1.0)
        layers.append((lay, ox, oy))
    X0 = min(o[1] for o in layers); Y0 = min(o[2] for o in layers); X1 = max(o[1] + o[0].width for o in layers); Y1 = max(o[2] + o[0].height for o in layers)
    full = Image.new("RGBA", (X1 - X0, Y1 - Y0), (0, 0, 0, 0))
    for lay, ox, oy in layers:
        full.alpha_composite(lay, (ox - X0, oy - Y0))
    e = mf.add("headline", "Cover lines: Air Pirate Nathan Zachary Wows Audiences!", "text", full, X0, Y0, 20,
               {"type": "retyped", "method": "Times, gold, set along the baselines/extents measured on the original's gold strokes (20 deg tilt); the page below is wiped of gold inside the box", "tool": "tools/parts_mag3.py headline"}, "problem")
    e["wipe_gold"] = [[[int(X0 - 6), int(Y0 - 6)], [int(X1 + 6), int(Y0 - 6)], [int(X1 + 6), int(Y1 + 6)], [int(X0 - 6), int(Y1 + 6)]]]
    mf.save()


def portrait_source():
    return Image.fromarray(rgb_orig()).crop(PORT_BOX)


FACE_PROMPT = ("Edit the first image: keep exactly the same pose (facing the camera, no hands visible), the flying cap and goggles, the white scarf, jacket, framing, lighting and backdrop, changing only the man's "
               "face so that he is the same man as in the second image: same face shape, eyes, nose, mouth, jaw and ears. Do not copy the second image's cap or backdrop.")
FACE_SEED = 2


def portrait(force=False, seed=None):
    """pose from a Klein plate of the original thumbnail (hand to the face, goggles, scarf), the face of the ORIGINAL MAG1 portrait on it (multi-reference pass)"""
    W.mkdir(parents=True, exist_ok=True)
    src = portrait_source(); src.save(W / "portrait_src.png")
    if not (W / "portrait_plate.png").exists() or force:
        plate = mp.plate_up(src, PORT_PROMPT, seed or PORT_SEED, tag="mag3port", sigma_l=(1.5, 3), sigma_ab=(12, 24), keep=W)
        plate.save(W / "portrait_raw_plate.png")
        plate.resize((256, 256), Image.LANCZOS).save(W / "portrait_in.png")
        Image.open(mp.PARTS / "MAG1/work/man_src.png").convert("RGB").crop((42, 0, 118, 76)).resize((256, 256), Image.LANCZOS).save(W / "face_ref.png")   # the original MAG1 face, nothing upscaled by a model
        mp.gen_refs([W / "portrait_in.png", W / "face_ref.png"], W / "portrait_face.png", FACE_PROMPT, 256, 256, FACE_SEED)
        new = Image.open(W / "portrait_face.png").convert("RGB").resize(plate.size, Image.LANCZOS)
        new = mp.u8(mp.tu.colour_lock(mp.f32(new), mp.f32(plate), 6.0, 12.0))
        new.save(W / "portrait_plate.png")
    plate = Image.open(W / "portrait_plate.png").convert("RGB")
    ref = src.resize(plate.size, Image.LANCZOS)
    pm = mp.photo_mask(ref)                                                      # the tipped-on photo, from the original (paper is purple)
    plate = mp.sepia_project(plate, ref, 1.0, mask=pm.astype(bool))
    plate = mp.tone_match(plate, ref, pm.astype(np.float32), 0.4)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    a = cv2.GaussianBlur(cv2.erode(pm, k).astype(np.float32), (0, 0), 1.2)
    rgba = Image.merge("RGBA", (*plate.split(), Image.fromarray((a * 255 + .5).astype(np.uint8))))
    mf = mp.Manifest(C)
    mf.add("portrait", "Small portrait photo (Nathan Zachary in cap, goggles and scarf)", "figure", rgba, PORT_BOX[0] * 4, PORT_BOX[1] * 4, 10,
           {"type": "upscaled", "method": f"the thumbnail alone through FLUX Klein 2 x 2x (seed {PORT_SEED}) for pose, goggles and scarf; face re-matched to the ORIGINAL MAG1 portrait (multi-reference pass, seed {FACE_SEED}); sepia projection, tone match; alpha = the photo's outline from the original",
            "tool": "tools/parts_mag3.py portrait", "orig_box": list(PORT_BOX)}, "problem")


def badge():
    """the red '13¢' badge: the redraw's reads '13c'. The original's own pixels (crisp), clipped to the circle"""
    mf = mp.Manifest(C)
    import math
    box = (203, 15, 238, 48); poly = [(box[0] + 17.5 + 17.2 * math.cos(a / 24 * 2 * math.pi), box[1] + 16.8 + 16.6 * math.sin(a / 24 * 2 * math.pi)) for a in range(24)]
    im, x, y = mp.box_part(C, box, feather=3, amount=1.2, radius=1.4, steep=0.6, poly=poly, rgb=mp.orig_ext_rgb(C))
    mf.add("badge", "13\u00a2 price badge", "emblem", im, x, y, 30, {"type": "orig-crop", "method": "the original's own pixels (the cent sign is right there; the redraw turned it into a c): Lanczos 4x, unsharp, edge steepening, clipped to the circle", "tool": "mag_parts.box_part", "orig_box": list(box)}, "problem")


ICON_ZONES = {"cards": (196, 203, 258, 243), "star": (216, 235, 244, 262), "skull": (216, 258, 246, 292)}


def icons():
    import mag_icons as mi
    rgb = mp.opaque_on_grey(mp.SCRAPBOOK / "SB_11_01_MAG3.PNG")
    mf = mp.Manifest(C); placed = []
    for n, zone in ICON_ZONES.items():
        fit = mi.fit_icon(rgb, n, zone)
        r, x, y, info = mi.icon_part(rgb, n, fit)
        placed.append((r, x, y))
        r = mp.clip_to_fold(r, C, x, y)
        mf.add(n, {"cards": "Playing-card fan icon", "star": "Gold star icon", "skull": "Skull and crossbones icon"}[n], "emblem", r, x, y, 30,
               {"type": "vector", "method": f"shared vector icon (tools/mag_icons.py), fitted to the original by template match (scale {info['scale']:.2f}, angle {info['angle']}, NCC {info['score']:.2f}), tinted to this cover's darker colours",
                "tool": "tools/mag_icons.py", "shared": f"_shared/{n}.png", "orig_box": list(zone)}, "problem")
    im, x, y = mp.backdrop_patch(C, placed, grow=7)
    mf.add("icon_backdrop", "Backdrop under the icons (original's, old icons inpainted out)", "decor", im, x, y, 25,
           {"type": "orig-crop", "method": "the original's own black cloth with the old icons inpainted out at original resolution, Lanczos 4x", "tool": "mag_parts.backdrop_patch"}, "problem")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["all"]
    for fn in (portrait, plane, headline, icons, badge):
        if "all" in todo or fn.__name__ in todo:
            fn("force" in todo) if fn.__name__ in ("portrait", "plane") else fn()
    mp.compose(C, str(D / "composed.png"))
