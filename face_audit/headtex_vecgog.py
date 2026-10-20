"""headtex_vecgog.py: clean vector drawing of the goggles used as the klein reference for the goggle part -> headtex/part_gogv_ref.png (crop 344..680 x 335..495, 3x).
The traced goggle outline of the reference (left lens) is smoothed with a low-order Fourier fit (no pixel jaggies), mirrored about x=512; padding, rim and lens are exact inward offsets of it."""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_feat as FT
D = FT.D; CROP = (344, 335, 680, 495); K = 3; S = 2; HARM = 5
def smooth_outline():
    rgb = np.array(Image.open(D / "v_cphead_front.png").convert("RGB")); m = FT.masks(rgb)["goggles_dark"].copy(); m[:, :372] = False; m[:, 507:] = False; m[:342] = False; m[488:] = False
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); c = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]; t = np.linspace(0, d[-1], 512, endpoint=False); p = np.c_[np.interp(t, d, c[:, 0]), np.interp(t, d, c[:, 1])]
    z = p[:, 0] + 1j * p[:, 1]; Z = np.fft.fft(z); keep = np.zeros_like(Z); keep[:HARM + 1] = 1; keep[-HARM:] = 1; zs = np.fft.ifft(Z * keep)
    return np.c_[zs.real, zs.imag], rgb
def draw():
    left, rgb = smooth_outline(); x0, y0, x1, y1 = CROP; Wc, Hc = (x1 - x0) * K * S, (y1 - y0) * K * S
    P = lambda pts: np.round((pts - [x0, y0]) * K * S).astype(np.int32)
    mirror = left * [-1, 1] + [1024, 0]                                                                       # about x = 512
    Ml, Mr = np.zeros((Hc, Wc), np.uint8), np.zeros((Hc, Wc), np.uint8)
    cv2.fillPoly(Ml, [P(left)], 255, cv2.LINE_AA); cv2.fillPoly(Mr, [P(mirror)], 255, cv2.LINE_AA)
    br = np.zeros((Hc, Wc), np.uint8); cv2.rectangle(br, tuple(P(np.array([505.0, 342.0]))), tuple(P(np.array([519.0, 440.0]))), 255, -1)    # bridge between the lenses
    M = cv2.bitwise_or(cv2.bitwise_or(Ml, Mr), br)
    er = lambda m_, px: cv2.erode(m_, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1)))
    skin_lo = rgb[500:520, 380:640].reshape(-1, 3).mean(0); skin_hi = rgb[305:325, 400:620].reshape(-1, 3).mean(0); ys = np.linspace(0, 1, Hc)[:, None, None]
    out = ((skin_hi * (1 - ys) + skin_lo * ys) * np.ones((1, Wc, 1))).astype(np.uint8)
    cy = int((410 - y0) * K * S)
    for xs_, xe in ((0, int((372 - x0) * K * S)), (int((652 - x0) * K * S), Wc)): cv2.rectangle(out, (xs_, cy - 40), (xe, cy + 40), (38, 30, 28), -1)   # side straps
    out[M > 0] = (26, 22, 21)                                                                                  # leather padding
    rim = cv2.bitwise_or(er(Ml, 6 * 6), er(Mr, 6 * 6)); out[rim > 0] = (150, 154, 158)                                                                       # steel rim (6 ref px in)
    lens = cv2.bitwise_or(er(Ml, 6 * 11), er(Mr, 6 * 11)) > 0; grad = np.linspace(1.0, 0.55, Hc)[:, None, None]; out[lens] = (np.array([46, 62, 68]) * grad * np.ones((1, Wc, 1)))[lens].astype(np.uint8)
    out = cv2.resize(out, (Wc // S, Hc // S), interpolation=cv2.INTER_AREA); Image.fromarray(out).save(D / "part_gogv_ref.png")
if __name__ == "__main__": draw()
