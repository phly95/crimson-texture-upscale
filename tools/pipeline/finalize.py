"""finalize.py UNIT_DIR SEED [SUBDIR]: turn a klein redraw (RGB on a green screen) into the final RGBA picture  klein_s<seed>_rgba.png.

v2: the redraw's OWN crisp outline (chroma key) is the alpha, because the original alpha is a low-resolution, blurry silhouette that would fuzz a crisp redraw.
The original alpha is used only for genuinely translucent INTERIOR patches (windows, glass, thin cloth): big coherent partly transparent areas well inside the shape.
Where the original is solid, the redraw is forced opaque (a green leak there is repainted, never a hole).  The green screen is removed from the colours at soft edges
(unmixing).  Fine foliage (the redraw has far more edge detail than the original alpha) keeps the plain chroma key, no forcing.  Opaque pictures are left alone."""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DETAIL_MAX = 1.6
IOU_MIN = 0.60
G = np.array([0.0, 255.0, 0.0], np.float32)


def _unmix(rgb, a, kk, lo=0.03, hi=0.97):
    """pixel = fg*a + green*(1-a)  ->  fg, where the redraw is tinted by the screen and the coverage is partial"""
    part = (a > lo) & (a < hi) & (kk > 8)
    if not part.any():
        return rgb
    fg = (rgb - G * (1 - a[..., None])) / np.maximum(a[..., None], 0.2)
    w = 0.6 * np.clip((kk - 8) / 30, 0, 1)[..., None] * part[..., None]
    return np.clip(rgb * (1 - w) + fg * w, 0, 255)


FIT_MIN = 1.08                                                     # smallest zoom worth applying


def _bbox(a, t=0.5):
    ys, xs = np.where(a > t)
    if not len(ys):
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _zoom_to_fill(d, rgb, alpha, ao):
    """A cut-out whose original runs edge to edge (a frame, a border, a tile) must keep that size: when the redraw drew it smaller, zoom it up (uniformly, never past the frame)
    around the original's centre.  Units listed in pipeline/fit_skip.json keep the picture the reviewer approved."""
    H, W = alpha.shape
    ob, rb = _bbox(ao), _bbox(alpha)
    if ob is None or rb is None:
        return rgb, alpha, None
    ow, oh = (ob[2] - ob[0]) / W, (ob[3] - ob[1]) / H
    rw, rh = (rb[2] - rb[0]) / W, (rb[3] - rb[1]) / H
    if ow < 0.94 or oh < 0.94:
        return rgb, alpha, None
    s = min(ow / rw, oh / rh)
    if s < FIT_MIN:
        return rgb, alpha, None
    try:
        import json
        skip = json.loads((Path(d).parent.parent / "fit_skip.json").read_text())
    except Exception:                                              # noqa: BLE001
        skip = []
    if Path(d).name in skip:
        return rgb, alpha, None
    cx, cy = (rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2
    tx, ty = (ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2
    M = np.array([[s, 0, tx - s * cx], [0, s, ty - s * cy]], np.float32)
    pm = np.dstack([np.clip(rgb, 0, 255) * alpha[..., None], alpha]).astype(np.float32)
    w = cv2.warpAffine(pm, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a2 = np.clip(w[..., 3], 0, 1)
    c2 = w[..., :3] / np.maximum(a2[..., None], 1e-3)
    return c2, a2, round(float(s), 2)


def finalize(d, seed, sub=None):
    d = Path(d)
    out_dir = d / sub if sub else d
    cm = out_dir / f"klein_s{seed}_cm.png"
    src = cm if cm.exists() else out_dir / f"klein_s{seed}.png"
    orig = Image.open(d / "orig.png").convert("RGBA")
    ao_small = np.asarray(orig.getchannel("A")).astype(np.float32) / 255
    if (ao_small < 0.98).mean() < 0.005:                          # nothing transparent in the original: opaque picture, no alpha needed
        return None
    k = np.asarray(Image.open(src).convert("RGB")).astype(np.float32)
    H, W = k.shape[:2]
    S = max(H, W)
    ao = np.clip(np.asarray(orig.getchannel("A").resize((W, H), Image.BICUBIC)).astype(np.float32) / 255, 0, 1)
    kk = k[..., 1] - np.maximum(k[..., 0], k[..., 2])             # greenness
    ak = np.clip((110 - kk) / 65, 0, 1); ak[kk > 110] = 0           # the redraw's own coverage
    bg_hard = (kk > 70) & (k[..., 1] > 150)
    inter = ((ao >= 0.5) & ~bg_hard).sum(); uni = ((ao >= 0.5) | ~bg_hard).sum()
    iou = float(inter / max(1, uni))
    tv = lambda a: float(np.abs(np.diff(a, axis=0)).sum() + np.abs(np.diff(a, axis=1)).sum())      # noqa: E731
    ratio = tv(ak) / max(tv(ao), 1.0)
    detailed = ratio > DETAIL_MAX and iou < 0.97
    hand = False
    try:                                                           # hand-edited pictures (a letter removed, ...): the original's silhouette no longer applies, key by colour only
        import json
        if d.name in json.loads((d.parent.parent / "hand_edited.json").read_text()):
            detailed = True; hand = True
    except Exception:                                              # noqa: BLE001
        pass
    rgb = k.copy()
    info = {"iou": round(iou, 3), "detail": round(ratio, 2)}
    if iou < IOU_MIN or detailed:                                  # a different shape / fine foliage: plain chroma key
        # only trust "greenish" near the real screen: a green subject (an alien, leaves) far from any screen pixel is picture, not background
        band = cv2.dilate((kk > 150).astype(np.uint8), np.ones((2 * max(6, int(round(0.012 * S))) + 1,) * 2, np.uint8)) > 0
        ak = np.where(band | (kk > 150), ak, 1.0).astype(np.float32)
        rgb = _unmix(k, ak, kk)
        rgb[..., 1] = np.where((kk > 45) & band, np.minimum(rgb[..., 1], np.maximum(rgb[..., 0], rgb[..., 2])), rgb[..., 1])
        alpha = ak
        if hand:                                                   # keep the faint stray marks outside the sticker out of the picture
            alpha = alpha * (cv2.dilate((ao > 0.05).astype(np.uint8), np.ones((2 * int(round(0.03 * S)) + 1,) * 2, np.uint8)) > 0)
        if iou >= 0.75 and not hand:                                            # a similar shape: green screen drawn INSIDE the solid original is a mistake (the picture is dark olive, brown ...), not a hole
            oc = np.asarray(orig.convert("RGB").resize((W, H), Image.BICUBIC)).astype(np.float32)
            kk_o = oc[..., 1] - np.maximum(oc[..., 0], oc[..., 2])
            r_in = max(4, int(round(0.008 * S)))
            solid = cv2.erode((ao > 0.95).astype(np.uint8), np.ones((2 * r_in + 1, 2 * r_in + 1), np.uint8)) > 0
            ext = cv2.dilate((ao < 0.5).astype(np.uint8), np.ones((2 * max(6, int(round(0.015 * S))) + 1,) * 2, np.uint8)) > 0     # near the original's own empty area
            leak = (kk > 100) & solid & (kk_o < 70)
            n, lab, _, _ = cv2.connectedComponentsWithStats(leak.astype(np.uint8))
            fix = np.zeros((H, W), bool)
            for i in range(1, n):
                comp = lab == i
                if not (comp & ext).any():                         # not connected to the real empty area: the redraw left the screen colour in picture
                    fix |= comp
            if fix.any():
                fixd = cv2.dilate(fix.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                w = np.clip(cv2.GaussianBlur(fixd.astype(np.float32), (0, 0), 1.2), 0, 1)
                rgb = rgb * (1 - w[..., None]) + oc * w[..., None]
                alpha = np.maximum(alpha, w)
                info["screen_in_picture"] = round(float(fix.mean()), 4)
        info["alpha"] = "chroma key"
    else:
        r_in = max(4, int(round(0.008 * S)))
        zone_in = cv2.erode((ao > 0.95).astype(np.uint8), np.ones((2 * r_in + 1, 2 * r_in + 1), np.uint8)) > 0     # the original is solid here
        clipzone = cv2.dilate((ao > 0.05).astype(np.uint8), np.ones((2 * max(6, int(round(0.02 * S))) + 1,) * 2, np.uint8)) > 0
        alpha = np.where(zone_in, 1.0, ak) * clipzone
        # genuinely translucent interior patches of the original (windows, glass): large coherent partly transparent areas away from the outline
        part = ((ao > 0.1) & (ao < 0.9)).astype(np.uint8)
        part = cv2.morphologyEx(part, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        r_e = max(6, int(round(0.015 * S)))
        far = cv2.erode((ao > 0.02).astype(np.uint8), np.ones((2 * r_e + 1, 2 * r_e + 1), np.uint8)) > 0
        T = np.zeros((H, W), np.float32)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(part & far.astype(np.uint8))
        solid = ao > 0.9
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= 0.003 * H * W:
                comp = lab == i
                ring = (cv2.dilate(comp.astype(np.uint8), np.ones((2 * r_e + 1, 2 * r_e + 1), np.uint8)) > 0) & ~comp
                if ring.any() and solid[ring].mean() >= 0.6:            # enclosed by solid picture (a window), not the wide feather around an outline
                    T[comp] = 1
        if T.any():
            Tw = np.clip(cv2.GaussianBlur(T, (0, 0), 2.5), 0, 1)
            alpha = alpha * (1 - Tw) + ao * Tw
            zone_in &= (Tw < 0.05)
        info["translucent_patches"] = bool(T.any())
        # colours: unmix soft edges; a green leak inside the solid zone (or a strong green tint off the original's colour) is repainted
        rgb = _unmix(k, alpha, kk)
        oc = np.asarray(orig.convert("RGB").resize((W, H), Image.BICUBIC)).astype(np.float32)
        kk_o = oc[..., 1] - np.maximum(oc[..., 0], oc[..., 2])
        leak = (kk > 25) & (k[..., 1] > 90) & (kk_o < 12)                          # greenish in the redraw but not in the original: the screen showing through
        bad = ((bg_hard | leak) & zone_in) | (leak & (alpha > 0.5))
        if bad.any():
            nb, lb, sb, _ = cv2.connectedComponentsWithStats(bad.astype(np.uint8))
            bigb = np.zeros((H, W), bool)
            ext = cv2.dilate((ao < 0.5).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0       # the outside of the original silhouette
            for i in range(1, nb):
                comp = lb == i
                if sb[i, cv2.CC_STAT_AREA] > 0.004 * H * W or (comp & ext).any():              # large, or touching the outline (the redraw's outline is smaller): drop, do not invent
                    bigb |= comp
            if bigb.any():                                              # a large area the redraw left to the screen: the redraw did not cover it, drop the alpha there
                alpha = alpha * (1 - np.clip(cv2.GaussianBlur(bigb.astype(np.float32), (0, 0), 1.5), 0, 1))
            small = bad & ~bigb
            if small.any():                                             # small spots: repaint from the surroundings (the mask is grown so green-tinted neighbours are not used)
                hole = cv2.dilate(small.astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
                rgb = cv2.inpaint(np.clip(rgb, 0, 255).astype(np.uint8), hole, 5, cv2.INPAINT_TELEA).astype(np.float32)
                nb_ = (cv2.dilate(small.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0) & (kk_o < 12)
                rgb[..., 1] = np.where(nb_, np.minimum(rgb[..., 1], np.maximum(rgb[..., 0], rgb[..., 2])), rgb[..., 1])
        m = np.maximum(rgb[..., 0], rgb[..., 2])
        edge = (alpha > 0.01) & (alpha < 0.99) | (cv2.dilate((alpha < 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & (alpha > 0.01)
        rgb[..., 1] = np.where(edge & (kk > 0) & (kk_o < 12), np.minimum(rgb[..., 1], m), rgb[..., 1])
        info["alpha"] = "redraw outline" + (" + original translucency" if T.any() else "")
    rgb, alpha, fit = _zoom_to_fill(d, rgb, alpha, ao)
    if fit:
        info["fit"] = fit
    res = Image.fromarray(np.dstack([np.clip(rgb, 0, 255), np.clip(alpha, 0, 1) * 255]).round().astype(np.uint8), "RGBA")
    res.save(out_dir / f"klein_s{seed}_rgba.png")
    return info


if __name__ == "__main__":
    print(finalize(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
