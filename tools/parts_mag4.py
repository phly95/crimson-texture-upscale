"""parts_mag4.py: MAG4 (Hollywood Confidential, man in white shirt) rebuilt from parts -> pipeline/mag_parts/MAG4/  (python tools/parts_mag4.py [man|...|all])
 parts: page (approved redraw, base) | man (figure on its own, Klein 4x, face matched to the MAG1 portrait) | ..."""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mag_parts as mp

C = "MAG4"
D = mp.PARTS / C
W = D / "work"
BOX = (100, 68, 228, 308)                         # figure plate in original px (128x240)
SEED = 11
PROMPT = (mp.tu.BASE_PROMPT + " This is a vintage sepia studio photograph of a man in a leather flying cap with goggles, a loose white shirt, a holstered pistol, dark riding breeches "
          "tucked into tall leather boots, hands on hips.")
FACE_PROMPT = ("Edit the first image: keep exactly the same head size, position, head angle, flying cap, goggles, sepia lighting, film grain and plain backdrop, changing only the face: make it the "
               "same man as in the second image, with the same face shape, eyes, nose, mouth, jaw and ears. Do not copy the second image's cap, scale or backdrop.")
FACE_SEED = 4
HEAD = (165, 0, 365, 200)                         # head box in the 4x plate


def man_source():
    rgb = mp.opaque_on_grey(mp.SCRAPBOOK / "SB_11_01_MAG4.PNG").copy()
    f = rgb.astype(np.int16)
    m = ((f[..., 0] - f[..., 1] > 25) & (f[..., 0] > 90)).astype(np.uint8); m[:150] = 0; m[:, :90] = 0          # the red caption across the legs
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    c = mp.inpaint(rgb, m, 5, ns=True)
    f = c.astype(np.int16)
    wm = (f.min(-1) > 175).astype(np.uint8); wm[:, 128:] = 0; wm[:60] = 0; wm[130:] = 0                          # cover-line letters left of the man
    wm = cv2.dilate(wm, np.ones((3, 3), np.uint8))
    em = np.zeros(wm.shape, np.uint8); em[:160, 205:] = 1                                                      # emblem corner
    return Image.fromarray(mp.inpaint(c, np.maximum(wm, em), 4)).crop(BOX)


def head_fix(plate, seed=None):
    """the face of the MAG1 portrait on this body: one multi-reference Klein pass on the head box, blended back through an oval, tone-locked to the plate"""
    # identity reference = the ORIGINAL MAG1 portrait (cover lines removed, nothing upscaled by a model), not an approved upscale
    ref = Image.open(mp.PARTS / "MAG1/work/man_src.png").convert("RGB").crop((42, 0, 118, 76)).resize((256, 256), Image.LANCZOS)
    ref.save(W / "face_ref.png")
    hd = plate.crop(HEAD); hd.resize((256, 256), Image.LANCZOS).save(W / "head_in.png")
    mp.gen_refs([W / "head_in.png", W / "face_ref.png"], W / "head_out.png", FACE_PROMPT, 256, 256, seed or FACE_SEED)
    new = Image.open(W / "head_out.png").convert("RGB").resize(hd.size, Image.LANCZOS)
    new = mp.u8(mp.tu.colour_lock(mp.f32(new), mp.f32(hd), 5.0, 10.0))
    h, w = hd.height, hd.width
    m = np.zeros((h, w), np.float32); cv2.ellipse(m, (w // 2, int(h * 0.50)), (int(w * 0.30), int(h * 0.43)), 0, 0, 360, 1, -1)
    m = cv2.GaussianBlur(m, (0, 0), 9)[..., None]
    mixed = mp.f32(hd) * (1 - m) + mp.f32(new) * m
    out = plate.copy(); out.paste(mp.u8(mixed), HEAD[:2])
    return out


def man(force=False):
    W.mkdir(parents=True, exist_ok=True)
    src = man_source(); src.save(W / "man_src.png")
    if (W / "man_plate.png").exists() and not force:                    # the Klein passes are the slow, seeded part: reuse the saved plate unless 'force' is given
        plate = Image.open(W / "man_plate.png").convert("RGB")
        mt = mp.figure_matte(plate, (24, 0, 488, 960), fg_core=(180, 200, 330, 440))
        return _man_part(plate, mt)
    plate = mp.plate_up(src, PROMPT, SEED, tag="mag4man", sigma_l=(1.5, 3), sigma_ab=(12, 24), matte=((6, 0, 124, 240), (45, 50, 85, 110)), keep=W)
    plate.save(W / "man_raw_plate.png")
    plate = head_fix(plate)
    ref = src.resize(plate.size, Image.LANCZOS)
    mt = mp.figure_matte(plate, (24, 0, 488, 960), fg_core=(180, 200, 330, 440))
    plate = mp.tone_match(mp.sepia_project(plate, ref, 0.9), ref, mt, 0.5)
    plate.save(W / "man_plate.png")
    _man_part(plate, mt)


def _man_part(plate, mt):
    rgba = mp.figure_rgba(plate, None, None, blur=24, fade={"b": 90, "r": 30, "l": 40}, matte=mt, grow_xy=(110, 44))
    mf = mp.Manifest(C)
    mf.add("man", "Nathan Zachary full-length portrait", "figure", rgba, BOX[0] * 4, BOX[1] * 4, 10,
           {"type": "upscaled", "method": f"red caption, cover-line letters and emblem corner inpainted out of the original; the figure alone through FLUX Klein 2 x 2x (seed {SEED}); face re-matched to the ORIGINAL MAG1 portrait (multi-reference pass, seed {FACE_SEED}); edge-aware colour lock, sepia projection, partial tone match, GrabCut matte",
            "tool": "tools/parts_mag4.py (mag_parts.plate_up / gen_refs / figure_rgba)", "orig_box": list(BOX)}, "problem")


def caption():
    """the red pull-quote across the legs, set again crisp on top of the new figure (same geometry as mag_finish.MAG4_2)"""
    import mag_polish as pol, mag_sb_11_01_mag4 as ms
    from PIL import ImageFont
    st = "SB_11_01_MAG4"
    Hu, Wu = pol.upright_shape(st)
    f = ImageFont.truetype(ms.IMPACT, 100); f = ImageFont.truetype(ms.IMPACT, 100 * ms.CAP_W / f.getlength(ms.CAPTION[0]))
    def draw(d):
        for k, t in enumerate(ms.CAPTION):
            d.text((ms.CAP_X, ms.CAP_BASE + k * ms.CAP_PITCH), t, font=f, fill=255, anchor="ls")
    M = cv2.getRotationMatrix2D(ms.CAP_C, ms.ANG, 1.0)
    cap = pol.page_layer(st, cv2.warpAffine(ms.layer((Hu, Wu), draw), M, (Wu, Hu), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR))
    mf = mp.Manifest(C)
    im, x, y = mp.layer_part(cap, (205, 52, 42))
    mf.add("caption", "Red pull-quote: Rockets and bullets were flying everywhere, planes crashing into sets ... it was a disaster.", "text", im, x, y, 20,
           {"type": "retyped", "method": "IMPACT, set along the original's tilt", "tool": "tools/parts_mag4.py caption (mag_polish.page_layer geometry)"}, "problem")


ICON_ZONES = {"cards": (204, 72, 256, 120), "star": (224, 108, 246, 130), "skull": (218, 128, 246, 160)}
ICON_NAMES = {"cards": "Playing-card fan icon", "star": "Gold star icon", "skull": "Skull and crossbones icon"}


def icons():
    import mag_icons as mi
    rgb = mp.opaque_on_grey(mp.SCRAPBOOK / "SB_11_01_MAG4.PNG")
    mf = mp.Manifest(C)
    placed = []
    for n, zone in ICON_ZONES.items():
        fit = mi.fit_icon(rgb, n, zone)
        r, x, y, info = mi.icon_part(rgb, n, fit)
        placed.append((r, x, y))
        r = mp.clip_to_fold(r, C, x, y)
        mf.add(n, ICON_NAMES[n], "emblem", r, x, y, 30,
               {"type": "vector", "method": f"drawn as a vector (16x supersampled), fitted to the original by template match (scale {info['scale']:.2f}, angle {info['angle']}, NCC {info['score']:.2f}), tinted to the original's colour", "tool": "tools/mag_icons.py",
                "shared": f"_shared/{n}.png", "orig_box": list(zone)}, "problem")
    im, x, y = mp.backdrop_patch(C, placed, grow=7)
    mf.add("icon_backdrop", "Backdrop under the icons (original's, old icons inpainted out)", "decor", im, x, y, 25,
           {"type": "orig-crop", "method": "the original's own cloth/shadow with the old icons inpainted out at original resolution, Lanczos 4x", "tool": "mag_parts.backdrop_patch"}, "problem")


def run_compose():
    mp.compose(C, str(D / "composed.png"))


if __name__ == "__main__":
    todo = sys.argv[1:] or ["all"]
    for fn in (man, caption, icons):
        if "all" in todo or fn.__name__ in todo:
            fn(force="force" in todo) if fn is man else fn()
    run_compose()
