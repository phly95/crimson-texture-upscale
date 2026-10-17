"""fillholes.py: make everything enclosed by the subject opaque again.  For a green subject part (lens, leaf, gem) on a green screen the key drops it and leaves a hole inside a solid ring.
holes = alpha-transparent regions NOT connected to the picture border (4-connected flood from the border over alpha<thr); there alpha=1 and the colour is the redraw's own (screen-uncorrected) pixel."""
import numpy as np, cv2


def fill_holes(keyed_rgba, cm_rgb, thr=128, min_ring=1):
    a = keyed_rgba[..., 3]
    free = (a < thr).astype(np.uint8)
    H, W = free.shape
    ff = free.copy(); msk = np.zeros((H + 2, W + 2), np.uint8)
    for y, x in [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)] + [(0, i) for i in range(0, W, 16)] + [(H - 1, i) for i in range(0, W, 16)] + [(i, 0) for i in range(0, H, 16)] + [(i, W - 1) for i in range(0, H, 16)]:
        if ff[y, x] == 1:
            cv2.floodFill(ff, msk, (x, y), 2)
    holes = (free == 1) & (ff == 1)                            # transparent but not reachable from the border
    out = keyed_rgba.copy()
    soft = cv2.GaussianBlur(holes.astype(np.float32), (0, 0), 1.2)[..., None]
    out[..., :3] = (keyed_rgba[..., :3] * (1 - soft) + cm_rgb.astype(np.float32) * soft).round().astype(np.uint8)
    out[..., 3] = np.maximum(keyed_rgba[..., 3], (soft[..., 0] * 255).round().astype(np.uint8))
    return out, holes
