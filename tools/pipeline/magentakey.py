"""magentakey.py: soft key for a picture drawn on a flat MAGENTA (#FF00FF) screen - for subjects that are themselves green (green cars, foliage, green creatures) or cast shadows on green.
Inverting a magenta screen gives (0,255,0), so the soft green key (greenkey.green_key: coverage from the screen excess, colour un-mixing, rim despill) runs on the inverted picture and the colours are inverted back.
Cast shadows (black blended with the screen) become translucent neutral dark; green, tan and slate paint is untouched because it has no magenta excess."""
import numpy as np
from greenkey import green_key


def magenta_key(rgb, neutralize=True, green_subject=False):
    inv = 255.0 - rgb.astype(np.float32)
    arr, G, k0 = green_key(inv, neutralize=neutralize)
    out = arr.astype(np.float32).copy()
    out[..., :3] = 255.0 - out[..., :3]
    if green_subject:                                          # leaves/trees/green paint: in the rim band red and blue may not exceed green (kills the pink-purple fringe of the screen)
        import cv2
        a = out[..., 3]
        r = max(4, int(round(0.012 * max(a.shape))))
        band = cv2.dilate((a < 245).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        w = np.clip(cv2.GaussianBlur(band.astype(np.float32), (0, 0), r / 2.0), 0, 1)
        g = out[..., 1]
        for ch in (0, 2):
            out[..., ch] = out[..., ch] * (1 - w) + np.minimum(out[..., ch], g * 1.08 + 3) * w
    return np.clip(out, 0, 255).round().astype(np.uint8), G, k0
