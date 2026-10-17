"""gate.py: automatic acceptance test for texture outputs.

usage: gate.py <out_dir> [--quiet]        e.g. gate.py ~/workspace/csvm-hd/out/t0003

For every <uid>.png with a <uid>.json sidecar in the folder it checks:
  format      exact target size, RGB (no alpha) or RGBA (original had alpha)
  provenance  every library source id exists in assets/index.jsonl (all CC0)
  fidelity    "exact":    shrink the result back to the original size; SSIM and mean error must stay close,
                          i.e. the design (logo, glyph, nose art, gauge) is still the same design
              "material": low-frequency colour layout, mean colour and contrast must stay close and a
                          tileable original must stay tileable; fine detail is free to change
  alpha       same blend-vs-cutout class as the original (mirror of the engine's soft-alpha test),
              similar coverage; exact fidelity also needs matching silhouette
  detail gain output must be genuinely sharper than a plain smooth enlargement of the original
Writes gate.json and compare_<uid>.png (original | result shrunk | result 1:1 crop) into the folder.
Exit code 0 always; read gate.json ("pass" per unit).
"""
import json, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image, ImageDraw
import cv2
from skimage.metrics import structural_similarity
import texlab as T

TH = {
    "exact_ssim": 0.78, "exact_mae": 18.0, "exact_iou": 0.85,
    "mat_lowfreq_mae": 22.0, "mat_mean_shift": 14.0, "mat_contrast": (0.35, 3.0),
    "seam_floor": 1.6, "seam_mult": 1.3,
    "alpha_cov": 0.06, "gain": 1.15, "flat_std": 3.0, "orig_flat_ratio": 0.10,
    "grain_min_coh": 0.2, "grain_deg": 20.0, "grain_coh": (0.5, 1.8), "band": (0.3, 3.2),
}


def on_gray(im):
    """RGBA/RGB -> float RGB composited over mid-grey, plus visible-weight mask."""
    im = im.convert("RGBA")
    a = np.asarray(im, dtype=np.float32)
    al = a[..., 3:4] / 255.0
    rgb = a[..., :3] * al + 128.0 * (1 - al)
    return rgb, al[..., 0]


def sharp(rgb):
    g = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def ssim_gray(a, b):
    ga = cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(np.clip(b, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    m = min(ga.shape)
    win = 7 if m >= 7 else (m if m % 2 else m - 1)
    return float(structural_similarity(ga, gb, win_size=win, data_range=255))


def check(uid, path, meta):
    u = T.info(uid)
    problems, m = [], {}
    out = Image.open(path)
    tw, th = u["target"]
    if out.size != (tw, th):
        problems.append(f"size {out.size} != target {(tw, th)}")
        out = out.resize((tw, th), Image.LANCZOS)
    want_alpha = u["has_alpha"]
    if want_alpha and out.mode != "RGBA":
        problems.append("original has alpha, output is " + out.mode)
    if not want_alpha and out.mode != "RGB":
        problems.append("original is opaque, output must be RGB, got " + out.mode)
    for s in meta.get("sources", []):
        try:
            r = T.lib_rec(s)
            if r["license"] != "CC0-1.0":
                problems.append(f"source {s} licence {r['license']}")
        except KeyError:
            problems.append(f"unknown source {s}")

    o = T.orig(uid)
    fid = meta.get("fidelity", "exact")
    o_rgb, o_a = on_gray(o)
    out_rgba = out.convert("RGBA")
    big_rgb, big_a = on_gray(out_rgba)
    # shrink the grey-composited result by area averaging, the inverse of an enlargement
    small_rgb = cv2.resize(big_rgb, o.size, interpolation=cv2.INTER_AREA)
    small_a = cv2.resize(big_a, o.size, interpolation=cv2.INTER_AREA)

    mae = float(np.abs(small_rgb - o_rgb).mean())
    m["mae"] = round(mae, 2)
    vis = o_a > 0.1 if want_alpha else np.ones(o_a.shape, bool)
    if vis.sum() < 4:
        vis = np.ones(o_a.shape, bool)
    shift = np.abs(small_rgb[vis].mean(0) - o_rgb[vis].mean(0))
    m["mean_shift"] = round(float(shift.max()), 2)
    sig = max(o.size) / 16.0
    lf = float(np.abs(cv2.GaussianBlur(small_rgb, (0, 0), sig) - cv2.GaussianBlur(o_rgb, (0, 0), sig)).mean())
    m["lowfreq_mae"] = round(lf, 2)
    so, ss = float(o_rgb[vis].std()), float(small_rgb[vis].std())
    m["contrast_ratio"] = round(ss / (so + 1e-3), 2)

    if fid == "exact":
        s = ssim_gray(small_rgb, o_rgb)
        m["ssim"] = round(s, 3)
        if s < TH["exact_ssim"]:
            problems.append(f"exact: shrunk result differs from original (SSIM {s:.2f} < {TH['exact_ssim']})")
        if mae > TH["exact_mae"]:
            problems.append(f"exact: mean error {mae:.1f} > {TH['exact_mae']}")
    else:
        if lf > TH["mat_lowfreq_mae"]:
            problems.append(f"material: colour/brightness layout drifted (low-freq err {lf:.1f})")
        if shift.max() > TH["mat_mean_shift"]:
            problems.append(f"material: mean colour shifted by {shift.max():.1f}")
        lo, hi = TH["mat_contrast"]
        if so > TH["flat_std"] and not lo <= ss / (so + 1e-3) <= hi:
            problems.append(f"material: contrast ratio {ss / (so + 1e-3):.2f} outside {lo}-{hi}")
        # structure: grain direction/strength and the visible spatial scales must survive. A gravel
        # crop pasted over a streaky wood original matches colour and still fails this.
        oi = Image.fromarray(np.clip(o_rgb, 0, 255).astype(np.uint8))
        si = Image.fromarray(np.clip(small_rgb, 0, 255).astype(np.uint8))
        ao, co = T.grain_info(oi)
        au, cu = T.grain_info(si)
        m["grain"] = [round(ao), round(co, 2), round(au), round(cu, 2)]
        if co >= TH["grain_min_coh"]:
            dd = T.angle_diff(ao, au)
            if dd > TH["grain_deg"]:
                problems.append(f"material: grain direction wrong. Original streaks run {ao:.0f}deg "
                                f"(coherence {co:.2f}), result {au:.0f}deg (coherence {cu:.2f}); "
                                "rotate the source with align_grain or pick an asset with that grain")
            elif not TH["grain_coh"][0] <= cu / co <= TH["grain_coh"][1]:
                problems.append(f"material: grain strength off (coherence {cu:.2f} vs original {co:.2f})")
        eo, eu = T.band_energy(oi), T.band_energy(si)
        tot = sum(eo) + 1e-9
        bad = [(i, eu[i] / (eo[i] + 1e-9)) for i in range(3)
               if eo[i] / tot > 0.03 and not TH["band"][0] <= eu[i] / (eo[i] + 1e-9) <= TH["band"][1]]
        m["bands"] = [round(eu[i] / (eo[i] + 1e-9), 2) for i in range(3)]
        if bad:
            problems.append("material: detail at the original's own feature scales is off "
                            + ", ".join(f"band{i} x{r:.2f}" for i, r in bad)
                            + " (too smooth, too busy, or the wrong feature size; check inject sigma and crop zoom)")
        so_seam = T.seam_ratio(o.convert("RGB"))
        m["seam_orig"] = round(so_seam, 2)
        if so_seam <= TH["seam_floor"]:
            sr = T.seam_ratio(out.convert("RGB"))
            m["seam_out"] = round(sr, 2)
            if sr > max(TH["seam_floor"], so_seam * TH["seam_mult"]):
                problems.append(f"original tiles seamlessly, result has a visible seam (ratio {sr:.2f})")

    if want_alpha and out.mode == "RGBA":
        oa = np.asarray(o.getchannel("A"))
        ua = np.asarray(out.getchannel("A"))
        if T.soft_alpha(o) != T.soft_alpha(out):
            problems.append("alpha class changed (blend vs cutout): original soft=%s, result soft=%s"
                            % (T.soft_alpha(o), T.soft_alpha(out)))
        cov_o, cov_u = float((oa >= 128).mean()), float((ua >= 128).mean())
        m["alpha_cov"] = [round(cov_o, 3), round(cov_u, 3)]
        if abs(cov_o - cov_u) > TH["alpha_cov"]:
            problems.append(f"alpha coverage {cov_u:.2f} vs original {cov_o:.2f}")
        if fid == "exact":
            ref = np.asarray(o.getchannel("A").resize((tw, th), Image.BICUBIC)) >= 128
            cur = ua >= 128
            inter, union = float((ref & cur).sum()), float((ref | cur).sum())
            iou = inter / union if union else 1.0
            m["alpha_iou"] = round(iou, 3)
            if iou < TH["exact_iou"]:
                problems.append(f"exact: silhouette IoU {iou:.2f} < {TH['exact_iou']}")

    # detail gain over a plain smooth enlargement. A soft original (glow, flare, indicator lamp,
    # gradient, flat fill) has no high frequencies to gain: for those, reproducing the original
    # exactly IS the job, and demanding extra Laplacian energy would only reward invented grain.
    # Measured on the pilot: soft units sit at s0/std^2 <= 0.07, the least detailed real surface at 0.15.
    s0 = sharp(o_rgb)
    flat_orig = s0 / (so * so + 1e-6) < TH["orig_flat_ratio"]
    m["orig_flat"] = flat_orig
    plain, _ = on_gray(o.resize((tw, th), Image.LANCZOS))
    sp, sb = sharp(plain), sharp(big_rgb)
    m["detail_gain"] = round(sb / (sp + 1e-6), 2) if sp > 1e-6 else None
    if not flat_orig and so > TH["flat_std"] and sp > 1e-6 and sb / sp < TH["gain"]:
        problems.append(f"no real detail gain over a smooth enlargement ({sb / sp:.2f}x)")

    return {"uid": uid, "pass": not problems, "problems": problems, "metrics": m,
            "method": meta.get("method"), "fidelity": fid}


def compare_image(uid, path, dest):
    o = T.orig(uid, "RGBA")
    out = Image.open(path).convert("RGBA")
    P = 256

    def tile(im, label, crop=False):
        bg = Image.new("RGBA", (P, P), (255, 0, 255, 255))
        if crop:
            w, h = im.size
            s = min(P, w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        else:
            k = min(P / im.width, P / im.height)
            im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))),
                           Image.NEAREST if im.width < P / 2 else Image.LANCZOS)
        bg.alpha_composite(im, (0, 0))
        d = ImageDraw.Draw(bg)
        d.rectangle((0, P - 14, P, P), fill=(0, 0, 0, 255))
        d.text((3, P - 13), label, fill=(255, 255, 255, 255))
        return bg
    sheet = Image.new("RGBA", (P * 3 + 8, P), (30, 30, 36, 255))
    sheet.paste(tile(o, f"original {o.width}x{o.height} (nearest)"), (0, 0))
    sheet.paste(tile(out, f"result {out.width}x{out.height} (fit)"), (P + 4, 0))
    sheet.paste(tile(out, "result 1:1 centre crop", crop=True), (2 * P + 8, 0))
    sheet.convert("RGB").save(dest)


def main():
    d = Path(sys.argv[1]).expanduser()
    quiet = "--quiet" in sys.argv
    res = {}
    for p in sorted(d.glob("*.png")):
        uid = p.stem
        if uid.startswith("compare_"):
            continue
        mp = d / f"{uid}.json"
        if not mp.exists():
            continue                       # scratch image, not a deliverable (deliverables come from texlab.finish)
        meta = json.load(open(mp))
        try:
            res[uid] = check(uid, p, meta)
            compare_image(uid, p, d / f"compare_{uid}.png")
            # the PNG is for a human; this is what the agent can actually read (no image decoder here)
            with open(d / f"compare_{uid}.txt", "w") as fh:
                fh.write(T.compare_view(uid, p) + "\n")
                for pr in res[uid]["problems"]:
                    fh.write("PROBLEM: " + pr + "\n")
        except Exception as e:  # noqa
            res[uid] = {"uid": uid, "pass": False, "problems": [f"gate error: {e!r}"], "metrics": {}}
    for j in sorted(d.glob("*.json")):
        if j.name == "gate.json":
            continue
        meta = json.load(open(j))
        if meta.get("status") in ("esrgan_needed", "keep_original", "failed"):
            # keep_original is an accepted outcome: the pack ships the untouched original for that slot
            res.setdefault(meta["uid"], {"uid": meta["uid"], "pass": meta["status"] == "keep_original",
                                         "status": meta["status"],
                                         "problems": [], "notes": meta.get("notes", ""), "metrics": {}})
    json.dump(res, open(d / "gate.json", "w"), indent=1)
    if not quiet:
        for uid, r in res.items():
            tag = "PASS" if r["pass"] else (r.get("status", "FAIL").upper())
            print(f"{tag:14s} {uid}  {r.get('method') or ''}  {r['metrics']}")
            for pr in r["problems"]:
                print("      -", pr)
        print(f"{sum(r['pass'] for r in res.values())}/{len(res)} pass")


if __name__ == "__main__":
    main()
