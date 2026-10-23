"""parts_mag7.py: MAG7 (New York Weekly) rebuilt from parts -> pipeline/mag_parts/MAG7/  (python tools/parts_mag7.py [photo|caption|body|all] [force])
 parts: page (approved redraw, base) | photo (the cover photograph from the original, deskewed, Klein 4x on its own) | caption | body (retyped lettering over the photo)"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mag_parts as mp

C = "MAG7"
D = mp.PARTS / C
W = D / "work"
PHOTO = (75, 116, 291, 340)                       # photograph in the upright 1x frame (216 x 224)
PROMPT = (mp.tu.BASE_PROMPT + " This is a vintage sepia-toned photograph of a woman in a patterned coat and fur collar caught by a gust of wind, one gloved hand holding up a hat, a large flying boat or aircraft wing "
          "in the sky behind her, a city skyline with tall buildings, a railing and a wet street. Draw each arm, leg and the coat's flying hem exactly once: any blurred echo of a limb is just motion blur, not another limb.")
SEED = 72
PLANE_POLY = [(0, 0), (290, 0), (360, 45), (400, 85), (400, 135), (340, 215), (250, 300), (130, 335), (0, 300)]      # the wing in the photo plate (4x of PHOTO)
FD = Path.home() / ".local/share/fonts/winfonts"
CAPTION = dict(text="POLICE HARASS CITIZENS!", x0=345.0, x1=700.0, base=826.0, cap=21.0, font=str(FD / "arialbd.ttf"), ink=(226, 223, 218))
BODY_LINES = ["“Yeah, that pirate was just", "amazing,\" I mean, I'm constantly screaming", "at my stunt flyers to pull off maneuvers half as", "exciting. He dodged! He swooped! It was",
              "spectacular ... simply first-rate flying. It's like", "I always say: We should fire the union flyers", "and hire the pirates,\" said Kurt Edwards,",
              "director of Baroness. \"There were explosions,", "bullets zipping past my head, tremendous", "damage to the sets ... it was a total mess.", "I must admit, though, the footage we got was",
              "spectacular. Absolutely amazing. I've seen", "some fancy flying in my day, but this was", "breathtaking!\"", "Edwards, who many insiders", "believed was on the verge of losing his job,",
              "seems to be back on track (see page 2)."]
INDENT = {14}                                      # the second paragraph's first line is indented
BODY_X, BODY_INDENT, BODY_BASE, BODY_PITCH, BODY_SIZE, BODY_INK = 340.0, 20.0, 1014.0, 20.24, 17.5, (66, 60, 54)


def text_footprint():
    """where the cover's own lettering sits over the photograph, at 1x: the retyped caption and body layers (same geometry as the print) grown by 3 px"""
    cap = caption_layer(); body_ = body_layer()
    m = ((np.asarray(cap)[..., 3] > 10) | (np.asarray(body_)[..., 3] > 10)).astype(np.uint8)
    m = cv2.resize(m, (m.shape[1] // 4, m.shape[0] // 4), interpolation=cv2.INTER_MAX if hasattr(cv2, "INTER_MAX") else cv2.INTER_AREA) if False else cv2.resize(cv2.dilate(m, np.ones((9, 9), np.uint8)).astype(np.float32), (m.shape[1] // 4, m.shape[0] // 4), interpolation=cv2.INTER_AREA)
    return (m > 0.05).astype(np.uint8)


def photo_source():
    """the photograph from the deskewed original with the white caption and the dark body column inpainted out (only where the retyped lettering will sit), as a PIL RGB image"""
    u = mp.upright_1x(C); a = np.asarray(u).copy()
    fp = text_footprint()
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    med = cv2.medianBlur(g.astype(np.uint8), 9).astype(np.float32)
    m = (fp > 0) & (np.abs(g - med) > 5)
    m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) & fp
    return Image.fromarray(mp.inpaint(a, m, 3, ns=True)).crop(PHOTO)


def photo(force=False, seed=None):
    W.mkdir(parents=True, exist_ok=True)
    src = photo_source(); src.save(W / "photo_src.png")
    if not (W / "photo_plate.png").exists() or force:
        plate = mp.plate_up(src, PROMPT, seed or SEED, tag="mag7photo", sigma_l=(1.5, 3), sigma_ab=(12, 24), keep=W)
        plate.save(W / "photo_plate.png")
    plate = Image.open(W / "photo_plate.png").convert("RGB")
    ref = src.resize(plate.size, Image.LANCZOS)
    plate = mp.tone_match(mp.sepia_project(plate, ref, 0.9), ref, None, 0.4)
    # the aeroplane (a motion-blurred flying wing in the print): a redraw gave it cockpit windows, nacelles and a propeller. Its region is the original's own pixels instead
    dn = cv2.fastNlMeansDenoisingColored(np.asarray(src), None, 3, 3, 5, 15)
    flat = mp.crisp(Image.fromarray(dn), 4, 0.9, 1.6, 0.0).resize(plate.size, Image.LANCZOS)
    pm = np.zeros((plate.height, plate.width), np.float32)
    cv2.fillPoly(pm, [np.array(PLANE_POLY, np.int32)], 1.0); pm = cv2.GaussianBlur(pm, (0, 0), 14)[..., None]
    plate = mp.u8(mp.f32(plate) * (1 - pm) + mp.f32(flat) * pm)
    h, w = plate.height, plate.width
    m = np.zeros((h, w), np.float32); m[6:-6, 6:-6] = 1
    m = cv2.GaussianBlur(m, (0, 0), 3.0)
    m[:36] *= np.linspace(0, 1, 36)[:, None]                                    # the top edge meets the page's own sky: fade
    rgba = Image.merge("RGBA", (*plate.split(), Image.fromarray((m * 255 + .5).astype(np.uint8))))
    pg, x, y = mp.to_page(rgba, C, PHOTO[0] * 4, PHOTO[1] * 4)
    mf = mp.Manifest(C)
    mf.add("photo", "Cover photograph (woman in the gale, flying boat, skyline)", "photo", pg, x, y, 10,
           {"type": "upscaled", "method": f"the photograph cut from the deskewed original (caption and body text inpainted out), alone through FLUX Klein 2 x 2x (seed {SEED}), colour lock, sepia projection, tone match, rotated back to the page frame",
            "tool": "tools/parts_mag7.py photo", "orig_box_upright": list(PHOTO)}, "problem")


def _fit_line(text, font_path, size, x0, x1=None):
    f = ImageFont.truetype(font_path, size)
    return f


def upright_layer(items):
    """items: (text, x, base, font_path, size, ink, width|None) in the upright 4x frame -> RGBA layer (full upright canvas size)"""
    u = mp.upright_1x(C); Wc, Hc = u.width * 4, u.height * 4
    lay = Image.new("RGBA", (Wc, Hc), (0, 0, 0, 0))
    for text, x, base, fp, size, ink, width in items:
        f = ImageFont.truetype(fp, size)
        if width:
            f = ImageFont.truetype(fp, size * width / f.getlength(text))
        t = Image.new("L", (Wc, Hc), 0); ImageDraw.Draw(t).text((x, base), text, font=f, fill=255, anchor="ls")
        a = cv2.GaussianBlur(np.asarray(t, np.float32) / 255, (0, 0), 0.55)
        col = Image.new("RGBA", (Wc, Hc), tuple(ink) + (0,)); col.putalpha(Image.fromarray((a * 255).astype(np.uint8)))
        lay.alpha_composite(col)
    return lay


def caption_layer():
    c = CAPTION
    cap = c["cap"]
    f0 = ImageFont.truetype(c["font"], 200); hb = f0.getbbox("H"); size = 200 * cap / (hb[3] - hb[1])
    return upright_layer([(c["text"], c["x0"], c["base"], c["font"], size, c["ink"], c["x1"] - c["x0"])])


def body_layer():
    items = []
    for k, t in enumerate(BODY_LINES):
        items.append((t, BODY_X + (BODY_INDENT if k in INDENT else 0), BODY_BASE + k * BODY_PITCH, str(FD / "BELL.TTF"), BODY_SIZE, BODY_INK, None))
    return upright_layer(items)


def caption():
    lay = caption_layer()
    a = np.asarray(lay)[..., 3].astype(np.float32) / 255                        # soft dark halo so the white letters read over the buildings
    sh = cv2.GaussianBlur(a, (0, 0), 3.0) * 0.75
    base = np.zeros(a.shape + (4,), np.uint8); base[..., :3] = (20, 18, 16); base[..., 3] = (np.clip(sh, 0, 1) * 255).astype(np.uint8)
    base = Image.fromarray(base, "RGBA"); base.alpha_composite(lay); lay = base
    pg, x, y = mp.to_page(lay, C)
    mf = mp.Manifest(C)
    mf.add("caption", "White caption over the skyline: POLICE HARASS CITIZENS!", "text", pg, x, y, 20,
           {"type": "retyped", "method": "Arial Bold, white, set along the original's own line (upright frame, rotated to the page)", "tool": "tools/parts_mag7.py caption"}, "problem")


def target():
    """the target emblem (white disc, black arc, red ring, red line) drawn as a vector at 4x from the geometry measured on the original; where the logo and slogan overlap the disc the
    original's own brightness decides what is white"""
    k = 4; cx, cy, R = 46.3, 79.8, 22.6
    x0, y0, x1, y1 = 16, 52, 74, 106
    w, h = (x1 - x0) * k, (y1 - y0) * k; SS = 4
    big = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(big)
    def P(x, y): return ((x - x0) * k * SS, (y - y0) * k * SS)
    def circ(r): return [P(cx - r, cy - r), P(cx + r, cy + r)]
    d.ellipse(circ(R), fill=(250, 250, 250, 255))
    d.arc(circ(16.0), 122, 327, fill=(40, 40, 42, 255), width=int(0.95 * k * SS))
    d.ellipse(circ(11.2), outline=(196, 24, 28, 255), width=int(0.85 * k * SS))
    d.line([P(cx + .5, cy - .5), P(cx + 30, cy - 6.9)], fill=(196, 24, 28, 255), width=int(0.8 * k * SS))
    d.ellipse([P(cx - .6, cy - .6), P(cx + .6, cy + .6)], fill=(190, 20, 24, 255))
    img = big.resize((w, h), Image.LANCZOS)
    rgb, _ = mp.orig_rgb(C)
    o = np.asarray(rgb.crop((x0, y0, x1, y1)).resize((w, h), Image.LANCZOS)).astype(np.float32)
    lum = o.min(-1)
    white = np.clip((lum - 150) / 60.0, 0, 1)                                       # the disc is white in the original; logo, slogan and dark strip are not
    a = np.asarray(img).astype(np.float32); alpha = a[..., 3] / 255
    red = (a[..., 0] - a[..., 2] > 60)
    ink = (a[..., 3] > 0) & (a[..., :3].max(-1) < 120)
    keep = np.where(ink | red, np.clip((lum - 120) / 60.0, 0, 1) * 0 + 1, white)     # arc / ring / line wherever they are drawn inside the disc (kept by the disc test below)
    disc = np.zeros((h, w), np.float32); cv2.circle(disc, (int((cx - x0) * k), int((cy - y0) * k)), int(R * k), 1, -1); disc = cv2.GaussianBlur(disc, (0, 0), 1.0)
    inner = np.zeros((h, w), np.float32); cv2.circle(inner, (int((cx - x0) * k), int((cy - y0) * k)), int((R - 3) * k), 1, -1)
    out_a = alpha * np.where(disc > 0.5, np.maximum.reduce([white, inner, ink * 1.0, red * 1.0]), 0)
    out = np.dstack([a[..., :3], out_a * 255]).clip(0, 255).astype(np.uint8)
    mf = mp.Manifest(C)
    mf.add("target", "Target emblem (white disc, arc, red ring and line)", "emblem", Image.fromarray(out, "RGBA"), x0 * k, y0 * k, 30,
           {"type": "vector", "method": "drawn as a vector (4x supersampled) from geometry measured on the original; the original's brightness decides where logo/slogan overlap the disc", "tool": "tools/parts_mag7.py target", "orig_box": [x0, y0, x1, y1]}, "problem")


def backing():
    """the print sets its pull-quote on a pale, soft patch of the street (light haze) and the caption on a lighter patch of the skyline; the redraw's street there is dark and mottled, so each text
    block gets a soft pale backing of the original's own mean colour under it (upright frame, rotated to the page)"""
    u = np.asarray(mp.upright_1x(C)).astype(np.float32)
    def patch(box, feather, alpha, colour_box):
        x0, y0, x1, y1 = box
        Wc, Hc = u.shape[1] * 4, u.shape[0] * 4
        m = np.zeros((Hc, Wc), np.float32); m[y0:y1, x0:x1] = 1
        m = cv2.GaussianBlur(m, (0, 0), feather) * alpha
        cx0, cy0, cx1, cy1 = colour_box
        col = np.percentile(u[cy0 // 4:cy1 // 4, cx0 // 4:cx1 // 4].reshape(-1, 3), 70, axis=0)
        lay = np.empty((Hc, Wc, 4), np.uint8); lay[..., :3] = col.astype(np.uint8); lay[..., 3] = (m * 255).astype(np.uint8)
        return Image.fromarray(lay, "RGBA")
    body_ = patch((330, 990, 700, 1420), 28, 0.62, (330, 1000, 690, 1400))
    cap = patch((335, 790, 710, 836), 8, 0.42, (335, 790, 710, 836))
    body_.alpha_composite(cap)
    pg, x, y = mp.to_page(body_, C)
    mf = mp.Manifest(C)
    mf.add("text_backing", "Pale patches under the caption and pull quote", "decor", pg, x, y, 15,
           {"type": "orig-colour", "method": "soft patches in the original's own pale street/sky colour (percentile of the original under each block), as the print has behind its text", "tool": "tools/parts_mag7.py backing"}, "problem")


def body():
    lay = body_layer()
    pg, x, y = mp.to_page(lay, C)
    mf = mp.Manifest(C)
    mf.add("body", "Pull quote column (Kurt Edwards) over the street", "text", pg, x, y, 21,
           {"type": "retyped", "method": "Bell MT at the print's measure, light ink (the print is pale and soft); wording as printed on the cover", "tool": "tools/parts_mag7.py body"}, "problem")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["all"]
    for fn in (photo, backing, caption, body, target):
        if "all" in todo or fn.__name__ in todo:
            fn(force="force" in todo) if fn is photo else fn()
    mp.compose(C, str(D / "composed.png"))
