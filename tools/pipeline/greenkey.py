"""greenkey.py: soft chroma key for a picture on a pure green screen (clouds, smoke, glass, fur ...).
Each pixel is  fg*a + green*(1-a).  For a roughly neutral/blue-grey foreground (g close to max(r,b)) the green excess  kk = g - max(r,b)  gives the coverage:
a = 1 - (kk-k0)/(G-k0); the screen is then un-mixed out of the colour so no green fringe is left.  Interior pixels with no green excess stay fully opaque."""
import numpy as np
from PIL import Image


def green_key(rgb, k0=None, protect=True, neutralize=True):
    """rgb: float array HxWx3 (0..255). returns (rgba uint8)"""
    H, W = rgb.shape[:2]
    corner = np.concatenate([rgb[:12, :12].reshape(-1, 3), rgb[:12, -12:].reshape(-1, 3), rgb[-12:, :12].reshape(-1, 3), rgb[-12:, -12:].reshape(-1, 3)])
    G = np.median(corner, axis=0)                                  # the screen colour as drawn
    kk = rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2])
    kG = float(G[1] - max(G[0], G[2]))
    if k0 is None:                                                 # foreground's own green bias: the 3rd percentile of kk deep inside = where the picture is clearly solid
        k0 = float(np.clip(np.percentile(kk[kk < 0.25 * kG], 90) if (kk < 0.25 * kG).any() else 0, 0, 25)) + 2
    a = np.clip(1 - (kk - k0) / max(kG - k0, 1), 0, 1)
    fg = (rgb - G[None, None, :] * (1 - a[..., None])) / np.maximum(a[..., None], 0.08)
    fg = np.clip(fg, 0, 255)
    # remaining green tint in the foreground colours at partial coverage: cap g at max(r,b)+small
    import cv2
    r = max(6, int(round(0.012 * max(H, W))))
    halo = cv2.dilate((a < 0.97).astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0       # the rim: its colours picked up green light from the screen
    w = np.clip(cv2.GaussianBlur(halo.astype(np.float32), (0, 0), r / 2.0), 0, 1)
    cap = np.maximum(fg[..., 0], fg[..., 2])
    fg[..., 1] = fg[..., 1] * (1 - w) + np.minimum(fg[..., 1], cap) * w
    lum = (0.3 * fg[..., 0] + 0.59 * fg[..., 1] + 0.11 * fg[..., 2])[..., None]            # and the rim's remaining teal/mint cast: pull toward a neutral blue-grey
    neutral = np.concatenate([lum * 0.99, lum, lum * 1.03], axis=2)
    ww = (0.75 * w)[..., None] * (1.0 if neutralize else 0.0)        # neutralize=False keeps saturated colours (red lettering, orange fire) at the rim
    fg = np.clip(fg * (1 - ww) + neutral * ww, 0, 255)
    return np.dstack([fg, a * 255]).round().astype(np.uint8), G, k0


if __name__ == "__main__":
    import sys
    im = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(np.float32)
    out, G, k0 = green_key(im)
    Image.fromarray(out, "RGBA").save(sys.argv[2]); print("screen", G, "k0", k0)
