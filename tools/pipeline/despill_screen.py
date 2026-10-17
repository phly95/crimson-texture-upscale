"""despill_screen.py: clean up residual green-screen leftovers in a finished keyed RGBA (not for subjects that are legitimately bright green):
 1. bright, strongly green pixels (g>110 and g-max(r,b)>35) cannot be wood/metal/cloth: they are screen leaking through -> made transparent (grown by 1 px where still tinted)
 2. a 5 px band next to transparency: green capped at max(r,b)+6 (fringe of light reflected from the screen)."""
import numpy as np, cv2


def despill(arr, strong_kk=35, strong_g=110, band=5, slack=6):
    a = arr.astype(np.float32).copy()
    rgb, al = a[..., :3], a[..., 3]
    kk = rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2])
    leak = ((kk > strong_kk) & (rgb[..., 1] > strong_g) & (al > 0)).astype(np.uint8)
    leak = cv2.dilate(leak, np.ones((3, 3), np.uint8)) & (kk > 18).astype(np.uint8)
    al = np.where(leak > 0, 0, al)
    near = cv2.dilate((al < 128).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))) > 0
    cap = np.maximum(rgb[..., 0], rgb[..., 2]) + slack
    rgb[..., 1] = np.where(near, np.minimum(rgb[..., 1], cap), rgb[..., 1])
    a[..., :3], a[..., 3] = rgb, al
    return np.clip(a, 0, 255).round().astype(np.uint8)
