"""fringe.py: edge colour contamination of a cut-out. For every partly transparent pixel (0.02 < alpha < 0.98) compare its colour with the colour of the nearest
solid pixel (alpha >= 0.98): a clean anti-aliased edge has the colour of the object next to it, a keyed edge keeps the screen's green or a dark olive/yellow rim.
fringe_de = mean CIE76 dE over the band, weighted by alpha (visible pixels count more).   venv/bin/python tools/fringe.py [FILE ...]"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color


def fringe_de(path_or_array):
    a = path_or_array if isinstance(path_or_array, np.ndarray) else np.asarray(Image.open(path_or_array).convert("RGBA"), np.float32) / 255
    al = a[..., 3]
    band, solid = (al > .02) & (al < .98), al >= .98
    if band.sum() < 20 or solid.sum() < 20:
        return 0.0
    idx = ndi.distance_transform_edt(~solid, return_distances=False, return_indices=True)
    near = a[..., :3][idx[0], idx[1]]
    de = color.deltaE_cie76(color.rgb2lab(a[..., :3]), color.rgb2lab(near))
    return float((de[band] * al[band]).sum() / al[band].sum())


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, round(fringe_de(p), 2))


def rim_shift(cand, orig):
    """colour shift of the outermost solid ring (the rim an AI redraw or a key leaves around a cut-out) against the original's: CIE76 dE of the rings' mean
    Lab, and the a*/b* (hue) part alone. The ring is 1 original pixel wide, i.e. `scale` pixels in the candidate."""
    def ring(a, px):
        al = a[..., 3]
        solid = al >= .98
        if solid.sum() < 20 or solid.all():
            return None
        d = ndi.distance_transform_edt(solid)
        m = solid & (d <= px)
        lab = color.rgb2lab(a[..., :3])[m]
        return lab.mean(0) if len(lab) > 5 else None
    c = cand if isinstance(cand, np.ndarray) else np.asarray(Image.open(cand).convert("RGBA"), np.float32) / 255
    o = orig if isinstance(orig, np.ndarray) else np.asarray(Image.open(orig).convert("RGBA"), np.float32) / 255
    scale = max(1, round(c.shape[1] / o.shape[1]))
    rc, ro = ring(c, scale), ring(o, 1)
    if rc is None or ro is None:
        return 0.0, 0.0
    return float(np.linalg.norm(rc - ro)), float(np.linalg.norm(rc[1:] - ro[1:]))
