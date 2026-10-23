"""parts_mag1.py: MAG1 (Air Action Weekly) rebuilt from parts -> pipeline/mag_parts/MAG1/  (python tools/parts_mag1.py [man|caption|aa|all]); work files in pipeline/mag_parts/MAG1/work/
 parts: page (approved redraw, base) | man (figure, Klein 4x on its own) | caption (script retyped) | aa_emblem (traced from the original)"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mag_parts as mp

C = "MAG1"
D = mp.PARTS / C
W = D / "work"
PROMPT = (mp.tu.BASE_PROMPT + " This is a vintage sepia studio photograph of a man in a dark leather flying cap with aviator goggles pushed up on his forehead, a loose white shirt with a long "
          "white scarf hanging round his neck, his hands on his hips and a dark belt, against a plain grey backdrop.")
BOX = (96, 136, 312, 400)                       # the figure plate in original px (216x264, both multiples of 8)
SEED = 21


def man_source():
    """the original with the cover lines, the script caption, the masthead strip sliver and the page edge/shadow removed, so the figure is upscaled on its own"""
    rgb, al = mp.orig_rgb(C); rgb = rgb.copy(); rgb = np.asarray(rgb).copy()
    # opaque_on_grey: outside the page is grey, as in the viewer
    a = np.asarray(Image.open(mp.SCRAPBOOK / "SB_00_00_MAG1.PNG").convert("RGBA")); alpha = a[..., 3]
    rgb = np.asarray(Image.alpha_composite(Image.new("RGBA", (a.shape[1], a.shape[0]), (110, 110, 110, 255)), Image.fromarray(a)).convert("RGB")).copy()
    yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]]
    tm = mp.strokes_mask(rgb, poly=[(40, 150), (150, 150), (150, 265), (40, 265)], k=9, th=22, grow=2)
    cap = mp.strokes_mask(rgb, poly=[(40, 350), (335, 325), (335, 410), (40, 410)], k=9, th=40, grow=2)
    strip = ((yy < (148 - 0.1025 * (xx - 73) + 3)) & (yy >= 120) & (xx >= 90) & (xx <= 330)).astype(np.uint8)
    outside = cv2.dilate((alpha < 128).astype(np.uint8), np.ones((5, 5), np.uint8))
    band = ((cv2.dilate(outside, np.ones((25, 25), np.uint8)) > 0) & (rgb.max(-1) < 60)).astype(np.uint8)
    clean = mp.inpaint(rgb, np.maximum.reduce([tm, cap, strip, outside, band]), 6)
    return Image.fromarray(clean).crop(BOX)


def man():
    W.mkdir(parents=True, exist_ok=True)
    src = man_source(); src.save(W / "man_src.png")
    plate = mp.plate_up(src, PROMPT, SEED, tag="mag1man", sigma_l=(1.5, 3), sigma_ab=(12, 24), matte=((8, 0, 208, 264), (80, 110, 170, 262)), keep=W)
    ref = src.resize(plate.size, Image.LANCZOS)
    mt = mp.figure_matte(plate, (32, 0, 832, 1056), fg_core=(320, 440, 680, 1048))
    plate = mp.tone_match(mp.sepia_project(plate, ref, 0.9), ref, mt, 0.5)
    plate.save(W / "man_plate.png")
    rgba = mp.figure_rgba(plate, None, None, fade={"b": 60}, matte=mt, cut=[[(628, 880), (864, 880), (864, 1056), (628, 1056)]])
    mf = mp.Manifest(C)
    e = mf.add("man", "Nathan Zachary portrait", "figure", rgba, BOX[0] * 4, BOX[1] * 4, 10,
           {"type": "upscaled", "method": f"cover lines, caption and page edge inpainted out of the original; the figure alone through FLUX Klein 2 x 2x (seed {SEED}), edge-aware colour lock, sepia tone projection, partial tone match, GrabCut matte",
            "tool": "tools/parts_mag1.py (mag_parts.plate_up / figure_rgba)", "orig_box": list(BOX)}, "problem")
    # lower right of the figure: the old redraw's hip/trouser fragment is replaced by the original's own backdrop and page shadow (caption strokes removed), under the figure
    o = np.asarray(mp.orig_rgb(C)[0]).copy()
    o = mp.inpaint(o, mp.strokes_mask(o, poly=[(40, 350), (335, 325), (335, 410), (40, 410)], k=9, th=40, grow=2), 4)
    im, x, y = mp.box_part(C, (250, 358, 316, 404), feather=8, amount=0.6, rgb=Image.fromarray(o))
    mf.add("edge_patch", "Lower page edge right of the figure (original pixels)", "decor", im, x, y, 9, {"type": "orig-crop", "method": "the original's backdrop and page shadow, caption strokes inpainted, Lanczos 4x", "tool": "mag_parts.box_part", "orig_box": [250, 358, 316, 404]}, "problem")


def caption():
    mf = mp.Manifest(C)
    lay, x, y = mp.text_layer(tuple(mf.d["size"]), "Nathan Zachary: Gentleman Pirate!", 228, 1599, 1368, -0.118, 58, mp.WIN / "VINERITC.TTF", (238, 229, 204), shadow=0.5)
    mf.add("caption", "Caption: Nathan Zachary: Gentleman Pirate!", "text", lay, x, y, 20, {"type": "retyped", "method": "VINERITC (script) along the original's baseline, cream with soft shadow", "tool": "mag_parts.text_layer"}, "problem")


def aa():
    mf = mp.Manifest(C)
    im, x, y = mp.traced_part(C, (16, 150, 64, 198), [(24, 170), (48, 167), (58, 170), (58, 192), (36, 199), (24, 192)], steep=4, up_blur=2.2)
    e = mf.add("aa_emblem", "'AA' Air Action badge", "emblem", im, x, y, 30, {"type": "traced", "method": "two-tone trace of the original's own pixels (Lanczos 4x, steep sigmoid): no redraw, nothing invented", "tool": "mag_parts.traced_part", "orig_box": [16, 150, 64, 198]}, "problem")
    e["clears"] = [[[88, 672], [226, 666], [228, 780], [150, 800], [88, 772]]]; mf.save()


def badge():
    """the little crest at the top right of the masthead (cover text 'Aviation.' beside it): the original's own pixels, not the redraw's smudged copy"""
    mf = mp.Manifest(C)
    box = (284, 0, 314, 28)
    im, x, y = mp.box_part(C, box, feather=2, amount=1.1, radius=1.4, steep=0.5, rgb=mp.orig_ext_rgb(C))
    im = mp.alpha_clip(im, C, x, y)
    mf.add("crest", "Crest badge top right", "emblem", im, x, y, 31, {"type": "orig-crop", "method": "the original's own pixels: Lanczos 4x, unsharp, edge steepening, clipped to the paper", "tool": "mag_parts.box_part", "orig_box": list(box)}, "problem")


if __name__ == "__main__":
    mf = mp.Manifest(C)
    if not mf.d["size"]:
        raise SystemExit("run mag_parts init first")
    todo = sys.argv[1:] or ["all"]
    for fn in (man, caption, aa, badge):
        if "all" in todo or fn.__name__ in todo:
            fn()
    mp.compose(C, str(D / "composed.png"))
