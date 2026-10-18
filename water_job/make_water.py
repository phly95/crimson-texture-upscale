"""make_water.py GROUP [--amp A] [--sigma S] [--src K] [--out DIR]

Builds 16 seamless, coherent 512x512 HD water frames for one chapter group.
See NOTES.md for the method.  Run: venv/bin/python -I make_water.py C5
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

HERE = Path(__file__).resolve().parent
S = 8  # 64 -> 512


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32)


# ---------------------------------------------------------------- periodic helpers
def periodic_part(u):
    """Moisan periodic-plus-smooth decomposition per channel: returns the periodic component (no wrap seam)."""
    out = np.empty_like(u)
    for c in range(u.shape[2]):
        a = u[..., c].astype(np.float64)
        M, N = a.shape
        v = np.zeros_like(a)
        v[0, :] += a[-1, :] - a[0, :]; v[-1, :] += a[0, :] - a[-1, :]
        v[:, 0] += a[:, -1] - a[:, 0]; v[:, -1] += a[:, 0] - a[:, -1]
        q = np.arange(M)[:, None]; r = np.arange(N)[None, :]
        den = 2 * np.cos(2 * np.pi * q / M) + 2 * np.cos(2 * np.pi * r / N) - 4
        den[0, 0] = 1
        sh = np.fft.fft2(v) / den; sh[0, 0] = 0
        out[..., c] = (a - np.fft.ifft2(sh).real).astype(np.float32)
    return out


def blur_reflect(a, s):
    return np.stack([gaussian_filter(a[..., c], s, mode="reflect") for c in range(a.shape[2])], -1)


def periodic_blend(D, plateau=0.25):
    """Make a zero-mean detail image tile seamlessly WITHOUT cutting its structures: along each axis cross-fade (variance-preserving,
    cos/sin weights) the image with itself rolled by half a period, so the centre is the original and the wrap edge is the
    (continuous) rolled copy.  `plateau` = fraction of the half-period around the centre kept pure."""
    for ax in (0, 1):
        n = D.shape[ax]
        x = np.arange(n); d = np.abs(x - n / 2) / (n / 2)             # 0 at centre, 1 at the wrap edge
        t = np.clip((d - plateau) / (1 - plateau), 0, 1); t = t * t * (3 - 2 * t)
        th = t * np.pi / 2
        sh = [1] * D.ndim; sh[ax] = n
        D = np.cos(th).reshape(sh) * D + np.sin(th).reshape(sh) * np.roll(D, n // 2, ax)
    return D


def blur_wrap(a, s):
    if a.ndim == 3:
        return np.stack([gaussian_filter(a[..., c], s, mode="wrap") for c in range(a.shape[2])], -1)
    return gaussian_filter(a, s, mode="wrap")


def up_wrap(a, f=S, interp=cv2.INTER_CUBIC):
    """wrap-around upsample (pad, resize, crop) so the result tiles exactly."""
    h, w = a.shape[:2]; p = 8
    ap = np.pad(a, ((p, p), (p, p)) + ((0, 0),) * (a.ndim - 2), mode="wrap")
    r = cv2.resize(ap, (ap.shape[1] * f, ap.shape[0] * f), interpolation=interp)
    return r[p * f:(p + h) * f, p * f:(p + w) * f]


def box_down(a, f=S):
    return cv2.resize(a, (a.shape[1] // f, a.shape[0] // f), interpolation=cv2.INTER_AREA)


def warp(a, T):
    """out(x) = a(x - T(x)), T[...,0]=dx, T[...,1]=dy, wrap boundaries, bicubic."""
    H, W = a.shape[:2]
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    cy, cx = yy - T[..., 1], xx - T[..., 0]
    if a.ndim == 2:
        return map_coordinates(a, [cy, cx], order=3, mode="grid-wrap")
    return np.stack([map_coordinates(a[..., c], [cy, cx], order=3, mode="grid-wrap") for c in range(a.shape[2])], -1)


# ---------------------------------------------------------------- motion
def farneback(a, b, win=15, pad=24):
    A = np.pad(a, pad, mode="wrap"); B = np.pad(b, pad, mode="wrap")
    to8 = lambda x: np.clip(x * 40 + 128, 0, 255).astype(np.uint8)
    f = cv2.calcOpticalFlowFarneback(to8(A), to8(B), None, 0.5, 3, win, 5, 7, 1.5, 0)
    return f[pad:-pad, pad:-pad]


def motion_fields(orig, nf=8, fsig=1.5, flow_blur=2.0, win=15):
    """T[k] (k=0..nf): where frame 0's content is in frame k, in ORIGINAL pixels: content(x) of frame k = frame0(x - T_k(x))."""
    g = []
    for o in orig[:nf]:
        x = gaussian_filter(o.mean(2), fsig, mode="wrap"); g.append((x - x.mean()) / (x.std() + 1e-6))
    T = [np.zeros(g[0].shape + (2,), np.float32)]
    for k in range(nf):
        f = farneback(g[k], g[(k + 1) % nf], win)
        f = blur_wrap(f, flow_blur) if flow_blur else f
        Tn = f + warp(T[-1], f)          # T_{k+1}(x) = f(x) + T_k(x - f(x))
        T.append(Tn.astype(np.float32))
    return T


def lapvar(a):
    return float(cv2.Laplacian(cv2.cvtColor(np.clip(a, 0, 255).astype(np.float32) / 255, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group"); ap.add_argument("--amp", type=float, default=1.5, help="detail gain (relative to the source frame's own high-pass)")
    ap.add_argument("--sigma", type=float, default=4.0, help="high-pass sigma in HD px")
    ap.add_argument("--src", type=int, default=0, help="HD frame used as detail source")
    ap.add_argument("--out", default=None); ap.add_argument("--flowblur", type=float, default=2.0)
    ap.add_argument("--noflow", action="store_true")
    ap.add_argument("--mg", type=float, default=0.5, help="gain of the mid band (sigma..sigma2) of the detail layer")
    ap.add_argument("--sigma2", type=float, default=12.0)
    ap.add_argument("--bsig", type=float, default=None, help="blur of the upsampled original base (HD px); default = --sigma (complementary split)")
    ap.add_argument("--csig", type=float, default=None, help="smoothing of the residual correction toward the original; default: auto-tuned to tflicker <= --tf")
    ap.add_argument("--flat", type=float, default=0.8, help="strength (0..1) of local-energy flattening of the detail layer")
    ap.add_argument("--flatsig", type=float, default=40.0)
    ap.add_argument("--rolls", type=int, default=24, help="number of random rolls of the detail layer tried to minimise the worst wrap-seam score")
    ap.add_argument("--period", default="blend", choices=["blend", "moisan"])
    ap.add_argument("--tf", type=float, default=1.25)
    a = ap.parse_args()
    g = a.group
    O = HERE / "data" / g / "orig"; H = HERE / "data" / g / "hd_current"
    out = Path(a.out) if a.out else HERE / "out" / g
    out.mkdir(parents=True, exist_ok=True)
    orig = [load(O / f"wtr{k:05d}.png") for k in range(16)]
    nf = 8
    # sanity: frames k and k+8 identical?
    rep = max(np.abs(orig[k] - orig[k + 8]).max() for k in range(8))
    assert rep < 1, f"frames do not repeat with period 8 (max diff {rep})"

    # 1. one periodic detail layer from the approved HD frame
    src = load(H / f"wtr{a.src:05d}.png")
    if a.period == "moisan":
        per = periodic_part(src)
        G1 = blur_wrap(per, a.sigma)
        D = (per - G1) + a.mg * (G1 - blur_wrap(per, a.sigma2))
    else:  # band-split the non-periodic frame (reflect), then make the BAND tileable by self-cross-fade
        G1 = blur_reflect(src, a.sigma)
        D = (src - G1) + a.mg * (G1 - blur_reflect(src, a.sigma2))   # fine band at full gain + attenuated mid band
        D = periodic_blend(D)
    # remove any chroma bias: keep the AI's colour detail but zero-mean each channel
    D -= D.mean((0, 1), keepdims=True)
    if a.flat > 0:  # even out the local detail energy (periodic blur) so no region of the tile is special
        en = np.sqrt(blur_wrap((D ** 2).mean(2), a.flatsig)) + 1e-3
        D = D * ((en.mean() / en) ** a.flat)[..., None]
        D -= D.mean((0, 1), keepdims=True)
    D *= a.amp

    # 2. motion fields (orig px) -> HD px
    if a.noflow:
        T = [np.zeros((64, 64, 2), np.float32)] * (nf + 1)
    else:
        T = motion_fields(orig, nf, flow_blur=a.flowblur)
    Th = [np.stack([up_wrap(t[..., 0]), up_wrap(t[..., 1])], -1) * S for t in T]
    T8 = Th[nf]
    print("mean drift over loop (hd px):", T8.reshape(-1, 2).mean(0).round(1), "std", T8.reshape(-1, 2).std(0).round(1), flush=True)

    # 3. detail for each of the 8 unique frames: moving copy A and loop-closing copy B, variance-preserving crossfade
    def make_det(D):
        det = []
        for k in range(nf):
            w = k / nf
            A = warp(D, Th[k]); B = warp(D, Th[k] - T8)   # B(x)=D(x - T_k + T_8): closes onto D at k=8
            det.append(np.cos(np.pi * w / 2) * A + np.sin(np.pi * w / 2) * B)
        return det
    det = make_det(D)

    # 4. low-frequency base from the original, plus detail; partial back-projection toward the original
    #    (smoothed residual correction: fixes mean colour / large shapes exactly, csig sets how much of the
    #    detail layer's box-scale content is cancelled, which is what controls the check's tflicker).
    bsig = 3.0 if a.bsig is None else a.bsig
    base = [blur_wrap(up_wrap(orig[k]), bsig) for k in range(nf)]
    red = lambda x: np.round(np.clip(x, 0, 255))
    orig_ch = np.mean([np.abs(orig[(k + 1) % nf] / 255 - orig[k] / 255).mean() for k in range(nf)])

    def render(csig, quick=False):
        gain = np.ones(nf)

        def build(k):
            x = base[k] + gain[k] * det[k]
            for _ in range(3):
                x = x + blur_wrap(up_wrap(orig[k] - box_down(x)), csig)
            return x
        for it in range(1 if quick else 5):
            fr = [build(k) for k in range(nf)]
            lv = np.array([lapvar(f) for f in fr])
            gain = np.clip(gain * (np.exp(np.log(lv).mean()) / lv) ** 0.5, 0.3, 3)
        fr = [red(build(k)) for k in range(nf)]
        sm = [box_down(f / 255) for f in fr]
        tf = np.mean([np.abs(sm[(k + 1) % nf] - sm[k]).mean() for k in range(nf)]) / orig_ch
        return fr, tf, gain

    def seam_score(f):
        a_ = f / 255; r = []
        for ax in (0, 1):
            e = np.abs(np.take(a_, 0, ax) - np.take(a_, -1, ax)).mean(); i_ = np.abs(np.diff(a_, axis=ax)).mean()
            r.append(e / max(i_, 1e-5))
        return float(np.mean(r))

    if a.rolls > 0:
        # the periodic detail layer can be rolled freely: pick the roll whose worst frame has the lowest wrap-seam score
        rng = np.random.default_rng(1)
        best = None
        for n in range(a.rolls):
            sh = (0, 0) if n == 0 else tuple(int(v) for v in rng.integers(0, 512, 2))
            det = make_det(np.roll(D, sh, (0, 1)))
            fr_, _, _ = render(a.csig if a.csig else 2.0, quick=True)
            sc = max(seam_score(f) for f in fr_)
            if best is None or sc < best[0]: best = (sc, sh)
        print("roll search: best", best)
        det = make_det(np.roll(D, best[1], (0, 1)))

    if a.csig is not None:
        fr, tf, gain = render(a.csig)
    else:  # largest csig (most detail body) with tflicker <= target
        lo, hi = 0.8, 10.0
        fr, tf, gain = render(lo)
        for _ in range(7):
            mid = (lo * hi) ** 0.5
            f2, t2, g2 = render(mid)
            if t2 <= a.tf: lo, fr, tf, gain = mid, f2, t2, g2
            else: hi = mid
        a.csig = lo
    print(f"{g}: csig={a.csig:.2f} tflicker~{tf:.2f} gain={np.round(gain, 2)}")
    for k in range(16):
        Image.fromarray(fr[k % nf].astype(np.uint8)).save(out / f"wtr{k:05d}.png")


if __name__ == "__main__":
    main()
