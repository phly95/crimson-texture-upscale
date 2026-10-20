"""headtex_fin.py STEM SEED: align a headtex_gen redraw to the original (smooth optical-flow warp on the broad structure, then Lab colour lock) -> headtex/<stem>_s<seed>_fin.png + metrics"""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_gen as H
T = H.T
def lowcorr(a, b, s=6):
    g = lambda x: cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (0, 0), s).astype(np.float32)
    a, b = g(a), g(b); a -= a.mean(); b -= b.mean(); return float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum()))
def align(gen, ref, sig=4.0, smooth=20.0, clip=40.0):
    """warp gen onto ref; returns warped, flow"""
    f = lambda x: cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (0, 0), sig)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fl = dis.calc(f(ref), f(gen), None)                           # ref pixel p -> where it is in gen
    fl = cv2.GaussianBlur(fl, (0, 0), smooth); m = np.hypot(fl[..., 0], fl[..., 1]); fl *= np.minimum(1, clip / np.maximum(m, 1e-6))[..., None]
    h, w = ref.shape[:2]; gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(gen, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT), fl
def fin(stem, seed, sl=8.0, sab=24.0, **kw):
    D = H.ROOT / "face_audit/headtex"
    ref = np.array(Image.open(D / f"{stem}_ref.png").convert("RGB")); gen = np.array(Image.open(D / f"{stem}_s{seed}.png").convert("RGB").resize(ref.shape[1::-1], Image.LANCZOS))
    w, fl = align(gen, ref, **kw)
    out = np.clip(T.colour_lock(w / 255.0, ref / 255.0, sl, sab), 0, 1); out = (out * 255 + .5).astype(np.uint8)
    Image.fromarray(out).save(D / f"{stem}_s{seed}_fin.png")
    print(stem, seed, "lowcorr raw %.3f aligned %.3f final %.3f" % (lowcorr(gen, ref), lowcorr(w, ref), lowcorr(out, ref)), "flow mean %.1f max %.1f px" % (np.hypot(fl[..., 0], fl[..., 1]).mean(), np.hypot(fl[..., 0], fl[..., 1]).max()))
    return out, ref
if __name__ == "__main__":
    out, ref = fin(sys.argv[1], int(sys.argv[2]))
    ov = (ref * .5 + out * .5).astype(np.uint8)
    Image.fromarray(np.hstack([ref, out, ov])).resize((1536, 512 * ref.shape[0] // ref.shape[1] if False else 512)).save(H.ROOT / f"face_audit/headtex/ov_{sys.argv[1]}_s{sys.argv[2]}.png")
