"""tile_batch.py LIST.json IDX[,IDX...]|all: run the tiled FLUX 4x upscale unattended for pictures that hold no lettering (material only), one job per picture, and submit each
result as the newest custom version of its unit (it then shows in the swipe review). Uses the neutral crop hint the text jobs settled on ("redraw only what is in the crop").
Each entry: [uid, ui_src, w, h]; the per-picture sentence below says what the material is and that nothing may be added."""
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tile_job as tj
from PIL import Image

NEUTRAL = "This is a detail crop of a larger picture. Redraw only what is visible in this crop; never add anything that is not already in the crop. "
KEEP = " It contains no lettering of any kind: do not add any text, labels, numbers, logos or symbols. Do not add any creases, wrinkles, scratches, stains or wear that are not already there."
WHAT = {
    "loadframempt": "It shows a dark gunmetal panel with thin white rules and a copper-brown riveted border.",
    "loadframempt2": "It shows a dark gunmetal panel with thin white rules and a copper-brown riveted border.",
    "KB_BACKGROUND": "It shows riveted dark weathered metal panels and frames on a black background.",
    "VP_BACKGROUND": "It shows riveted dark weathered metal panels and frames on a black background.",
    "MP_GAMESBACKGROUND": "It shows a riveted metal wall panel with shadows and an illustrated pin-up skull painting.",
    "MP_LOBBY_BACKGROUND": "It shows a riveted brass-trimmed panel with cream paper-like fields and a faint illustration.",
    "MP_OPTIONSBACKGROUND": "It shows a riveted brass-trimmed panel with soft golden paper-like fields.",
    "MS_BACKGROUND2": "It shows a metal-framed blank white panel in a dim room corner with dark wood and olive cloth.",
    "OL_BACKGROUND": "It shows a clipboard with a brass clip, a blank blue-lined grey form sheet and a dark red wood table.",
    "PS_BACKGROUND": "It shows a clipboard with a brass clip, a blank blue-lined grey form sheet and a dark red wood table.",
    "SB_BACKGROUND": "It shows an open blank book with soft aged cream pages and an orange-brown leather cover on a dark red wood table.",
    "SB_BG_A": "It shows soft aged cream paper pages of a book and an orange-brown leather edge.",
    "SB_BG_G": "It shows a sheet of soft pink-orange aged paper beside a cream book page.",
    "SB_BG_Q": "It shows soft aged cream paper pages of a book and an orange-brown leather edge.",
}
GENERIC = {"SB_BG_": "It shows a sheet of paper lying on a cream book page next to an orange-brown leather edge."}


DARK = {"loadframempt", "loadframempt2", "KB_BACKGROUND", "VP_BACKGROUND"}      # near-black pictures whose faint scan/JPEG noise FLUX otherwise redraws as lines and grids


def preclean_dark(src, out):
    """denoise only the dark, low-contrast parts of the source (non-local means), so faint noise is not mistaken for structure; bright lines, rivets and plates are left alone"""
    import cv2, numpy as np
    im = cv2.cvtColor(np.asarray(Image.open(src).convert("RGB")), cv2.COLOR_RGB2BGR)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    near_bright = cv2.dilate((g > 0.30).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    mask = ((g < 0.30) & ~near_bright).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), 2.0)[..., None]
    den = cv2.fastNlMeansDenoisingColored(im, None, 14, 14, 7, 21).astype(np.float32)
    res = den * mask + im.astype(np.float32) * (1 - mask)
    Image.fromarray(cv2.cvtColor(res.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)).save(out)
    return out


def what(stem):
    if stem in WHAT:
        return WHAT[stem]
    for k, v in GENERIC.items():
        if stem.startswith(k):
            return v
    return "It shows a clean material surface."


def upscale_job(name, src, hint, stage_cb=None):
    """tiled 4x of SRC as job `name` (resumable); returns the path of the 4x picture"""
    d = tj.jd(name)
    out = ROOT / "tile_job" / f"{name}_4x.png"
    if out.exists() and (d / "s4" / "stage.png").exists():
        return out
    if not (d / "s2" / "grid.json").exists():
        ns = type("A", (), dict(name=name, src=str(src), hint=hint, sigma_l=4.0, sigma_ab=12.0))
        tj.new(ns)
    import json as _j
    for stage in ("s2", "s4"):
        gr = _j.loads((d / stage / "grid.json").read_text())
        for yi in range(len(gr["ys"])):
            for xi in range(len(gr["xs"])):
                if not (d / stage / f"out_{yi}{xi}.png").exists():
                    tj.gen(name, f"{stage}_{yi}{xi}")
        tj.assemble(name, stage)
        if stage == "s2":
            tj.cut_refs(name, "s4", Image.open(d / "s2" / "stage.png").convert("RGB"))
    Image.open(d / "s4" / "stage.png").save(out)
    return out


def with_alpha(out, src):
    """the redraw is RGB; a source with transparency gets its own alpha back (upscaled, Lanczos) and colour bled under the transparent part"""
    import numpy as np
    sp = Image.open(src)
    if "A" not in sp.convert("RGBA").getbands() or np.asarray(sp.convert("RGBA").getchannel("A")).min() > 250:
        return out
    im = Image.open(out).convert("RGB")
    a = sp.convert("RGBA").getchannel("A").resize(im.size, Image.LANCZOS)
    res = out.with_name(out.stem + "_rgba.png")
    Image.merge("RGBA", (*im.split(), a)).save(res)
    return res


def submit(uid, out, label="Tiled FLUX 4x (3200x2400)", src=None):
    if src is not None:
        out = with_alpha(Path(out), src)
    r = subprocess.run([str(ROOT / "venv/bin/python"), str(ROOT / "flux_fix/submit_fix.py"), uid, str(out), "--label", label], capture_output=True, text=True)
    return (r.stdout.strip()[-120:] + " " + r.stderr.strip()[-200:]).strip()


def run(entry):
    uid, src, w, h = entry
    stem = Path(src).stem
    d = tj.jd(stem)
    if (d / "submitted").exists():
        print("skip (done)", stem); return
    t0 = time.time()
    srcp = Path.home() / "Games/CSVM/extracted" / src
    if stem in DARK:
        srcp = preclean_dark(srcp, ROOT / "tile_job" / f"{stem}_preclean.png")
        stem_job = stem + "_pc"
    else:
        stem_job = stem
    out = upscale_job(stem_job, srcp, NEUTRAL + what(stem) + KEEP)
    res = submit(uid, out, src=Path.home() / "Games/CSVM/extracted" / src, label="Tiled FLUX 4x, noise cleaned first" if stem in DARK else "Tiled FLUX 4x (3200x2400)")
    print(stem, res, f"{time.time() - t0:.0f}s", flush=True)
    if "submitted as" in res or "already submitted" in res:
        d.mkdir(parents=True, exist_ok=True)
        (d / "submitted").write_text(time.strftime("%F %T"))


def run_v2(entry):
    """second attempt for pictures where the first batch invented forms/lines: neutral hint only, 'plain surface' sentences on featureless tiles, flat-area guard on the result"""
    import json as _j
    from suppress_invented import suppress
    uid, src, w, h = entry
    stem = Path(src).stem
    name = stem + "_v2"
    d = tj.jd(name)
    if (d / "submitted").exists():
        print("skip (done)", name); return
    t0 = time.time()
    srcp = Path.home() / "Games/CSVM/extracted" / src
    if stem in DARK:
        srcp = preclean_dark(srcp, ROOT / "tile_job" / f"{stem}_preclean.png")
    if not (d / "s2" / "grid.json").exists():
        tj.new(type("A", (), dict(name=name, src=str(srcp), hint=NEUTRAL + "It contains no lettering.", sigma_l=4.0, sigma_ab=12.0)))
        j = _j.loads((d / "job.json").read_text()); j["auto_flat"] = True; (d / "job.json").write_text(_j.dumps(j, indent=1))
    out = upscale_job(name, srcp, "")
    res_img, frac = suppress(Image.open(srcp), Image.open(out).convert("RGB"))
    g = ROOT / "tile_job" / f"{name}_4x_guard.png"; res_img.save(g)
    res = submit(uid, g, src=Path.home() / "Games/CSVM/extracted" / src, label="Tiled FLUX 4x, plain-surface prompts + flat-area guard")
    print(name, f"flat {100 * frac:.0f}%", res, f"{time.time() - t0:.0f}s", flush=True)
    if "submitted as" in res or "already submitted" in res:
        d.mkdir(parents=True, exist_ok=True); (d / "submitted").write_text(time.strftime("%F %T"))


if __name__ == "__main__":
    items = json.load(open(sys.argv[1]))
    sel = range(len(items)) if sys.argv[2] == "all" else [int(x) for x in sys.argv[2].split(",")]
    runner = run_v2 if len(sys.argv) > 3 and sys.argv[3] == "v2" else run
    for i in sel:
        try:
            runner(items[i])
        except Exception as e:                                              # noqa: BLE001
            print("FAILED", items[i][1], repr(e), flush=True)
