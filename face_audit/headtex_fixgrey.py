"""headtex_fixgrey.py IN OUT: paint the atlas texels that collapsed-UV (broken) triangles sample with helmet leather, and warm the grey highlight patch on the crown (womanhead of the woman mesh)"""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_render as HR
D = Path(__file__).resolve().parent / "headtex"
def fix(tex, model=1175, pack="C1/gamez.zip", name="womanhead"):
    H, W = tex.shape[:2]; k = W / 1024.0; k = W / 1024.0; V, T = HR.tris(model, pack); bad = set(HR.broken(V, T)); lab = cv2.cvtColor(tex, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
    # helmet leather reference colour: dark, warm-ish, inside the head drawing (central region of the atlas)
    reg = np.zeros((H, W), bool); reg[int(.15 * H):int(.55 * H), int(.15 * W):int(.5 * W)] = True; helm = reg & (L < 40) & (L > 12)
    hm = lab[helm].mean(0); print("helmet Lab", (hm * [100 / 255, 1, 1] - [0, 128, 128]).round(1))
    out = lab.copy(); hmf = hm + np.array([8.0, 4.0, 8.0], np.float32)                                              # fill a little warmer / lighter than the measured dark core, like the visible leather
    # 0) atlas background outside the drawn head (grey / pale glow around the outline): paint with helmet leather with a gentle vertical sheen, so no triangle can sample grey or glow
    chroma = np.hypot(a, b); cand = ((chroma < 20) & (L >= 22)).astype(np.uint8); nlab, lbl = cv2.connectedComponents(cand, connectivity=4)
    border = set(np.unique(np.r_[lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]])) - {0}; bg = np.isin(lbl, list(border)) & (np.bincount(lbl.ravel())[lbl] > 3000 * k * k)
    print('background share of atlas %.0f%%' % (100 * bg.mean()))
    wb = cv2.GaussianBlur(cv2.dilate(bg.astype(np.uint8), np.ones((int(5 * k), int(5 * k)), np.uint8)).astype(np.float32), (0, 0), 2.0 * k)[..., None]
    yy = np.linspace(0, 1, H)[:, None, None]; sheen = hmf * (0.92 + 0.35 * np.sin(np.pi * yy)) * np.ones((1, W, 1)); sheen[..., 1:] = hmf[1:]
    out = out * (1 - wb) + sheen * wb
    # 1) the back-of-head triangles whose UVs collapse onto the atlas margin (vertical strip + diagonal to the nape): helmet leather above, leather -> neck skin along the diagonal
    skin = np.load(D / "woman_skin_target.npy").astype(np.float32)
    fillm = np.zeros((H, W), np.uint8)
    for j in (3, 4, 5, 9, 10, 11, 12, 24):                                                                          # all back-of-head / nape collapsed strips: plain helmet leather
        for vi, uv, nm in [T[j]]: cv2.fillPoly(fillm, [np.round(np.array(uv) * [W, H]).astype(np.int32)], 255); cv2.polylines(fillm, [np.round(np.array(uv) * [W, H]).astype(np.int32)], True, 255, int(17 * k))
    wl = cv2.GaussianBlur((fillm > 0).astype(np.float32), (0, 0), 2.5 * k)[..., None]; out = out * (1 - wl) + hmf * wl
    # 2) grey highlight patch on the crown (triangles 13 / 18 area): low chroma, lighter than helmet -> warm leather sheen
    crown = np.zeros((H, W), np.uint8)
    for j, (vi, uv, nm) in enumerate(T):
        if nm == name and j in (13, 18): cv2.fillPoly(crown, [np.round(np.array(uv) * [W, H]).astype(np.int32)], 255)
    crown = cv2.dilate(crown, np.ones((int(25 * k), int(25 * k)), np.uint8)) > 0
    hl = crown & (np.hypot(a, b) < 16) & (L > 24); wh = cv2.GaussianBlur(hl.astype(np.float32), (0, 0), 6)[..., None]
    warm = out.copy(); warm[..., 0] = np.minimum(out[..., 0], hm[0] * 1.45); warm[..., 1] = hm[1] + 3; warm[..., 2] = hm[2] + 6                    # keep some sheen, helmet hue
    out = out * (1 - wh) + warm * wh
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
if __name__ == "__main__":
    Image.fromarray(fix(np.array(Image.open(sys.argv[1]).convert("RGB")))).save(sys.argv[2])
