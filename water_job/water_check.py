"""water_check.py GROUP [DIR]: checks a 16-frame water flipbook (HD frames in DIR, default data/GROUP/hd_current) against the original frames data/GROUP/orig.
Reports per frame and in summary (venv/bin/python -I water_check.py C3 out/C3):
  de0           mean Lab colour of frame 0 against the original's frame 0 (dE76; limit 3: the engine's enhanced sea takes frame 0's average colour)
  de_mean       worst per-frame mean-colour difference over all frames (limit 4)
  recon         each HD frame reduced (BOX) to the original's size against the original frame: mean dE (limit 8: the HD frame must be the same picture)
  seam          wrap-around seam score of each HD frame (edge difference / typical neighbour difference), limit max(1.35 x the original's, 1.3)
  tflicker      mean frame-to-frame change (reduced to the original size, wrap 15 -> 0) of the HD set divided by the original's: limit 0.7..1.5
  tdetail       frame-to-frame variation of the HD frames' sharpness (std/mean of Laplacian variance over the 16 frames): limit 0.25
Writes strip.png (the 16 frames in a row of a 4x4 grid) and diff_strip.png (|frame k+1 - frame k| for the original and the HD set) next to the check, and prints JSON. Exit 0 = all limits met."""
import json, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from skimage import color

HERE = Path(__file__).resolve().parent
g = sys.argv[1]
D = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "data" / g / "hd_current"
O = HERE / "data" / g / "orig"
names = [f"wtr{k:05d}.png" for k in range(16)]
orig = [np.asarray(Image.open(O / n).convert("RGB"), np.float32) / 255 for n in names]
hd = [np.asarray(Image.open(D / n).convert("RGB"), np.float32) / 255 for n in names]
small = [cv2.resize(h, (orig[0].shape[1], orig[0].shape[0]), interpolation=cv2.INTER_AREA) for h in hd]
lab = lambda a: color.rgb2lab(a)
mean_lab = lambda a: lab(a).reshape(-1, 3).mean(0)
de = lambda a, b: float(np.linalg.norm(a - b))


def seam(a):
    r = []
    for ax in (0, 1):
        e = np.abs(np.take(a, 0, ax) - np.take(a, -1, ax)).mean()
        i = np.abs(np.diff(a, axis=ax)).mean()
        r.append(e / max(i, 1e-5))
    return float(np.mean(r))


res = {"group": g, "dir": str(D)}
res["de0"] = round(de(mean_lab(hd[0]), mean_lab(orig[0])), 2)
res["de_mean_frames"] = [round(de(mean_lab(h), mean_lab(o)), 2) for h, o in zip(hd, orig)]
res["de_mean"] = max(res["de_mean_frames"])
res["recon"] = round(float(np.mean([color.deltaE_cie76(lab(s), lab(o)).mean() for s, o in zip(small, orig)])), 2)
so, sh = [seam(o) for o in orig], [seam(h) for h in hd]
res["seam_orig_mean"], res["seam_hd_max"] = round(float(np.mean(so)), 2), round(max(sh), 2)
res["seam_limit"] = round(max(1.35 * float(np.mean(so)), 1.3), 2)
ch = lambda fr: float(np.mean([np.abs(fr[(k + 1) % 16] - fr[k]).mean() for k in range(16)]))
res["tflicker"] = round(ch(small) / max(ch(orig), 1e-6), 2)
lv = [cv2.Laplacian(cv2.cvtColor(h, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var() for h in hd]
res["tdetail"] = round(float(np.std(lv) / np.mean(lv)), 3)
res["pass"] = bool(res["de0"] <= 3 and res["de_mean"] <= 4 and res["recon"] <= 8 and res["seam_hd_max"] <= res["seam_limit"] and 0.7 <= res["tflicker"] <= 1.5 and res["tdetail"] <= 0.25)
print(json.dumps(res))
T = 128
def grid(frames, cols=4):
    ims = [Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)).resize((T, T), Image.LANCZOS) for f in frames]
    s = Image.new("RGB", (cols * (T + 2), (len(ims) // cols) * (T + 2)))
    for i, im in enumerate(ims):
        s.paste(im, ((i % cols) * (T + 2), (i // cols) * (T + 2)))
    return s
out = D if D != HERE / "data" / g / "hd_current" else HERE
a, b = grid(orig), grid(hd)
sh = Image.new("RGB", (a.width * 2 + 10, a.height)); sh.paste(a, (0, 0)); sh.paste(b, (a.width + 10, 0)); sh.save(out / "strip.png")
dd = lambda fr: [np.clip(np.abs(fr[(k + 1) % 16] - fr[k]) * 4, 0, 1) for k in range(16)]
a, b = grid(dd(orig)), grid(dd([cv2.resize(h, (64, 64), interpolation=cv2.INTER_AREA) for h in hd]))
sh = Image.new("RGB", (a.width * 2 + 10, a.height)); sh.paste(a, (0, 0)); sh.paste(b, (a.width + 10, 0)); sh.save(out / "diff_strip.png")
sys.exit(0 if res["pass"] else 1)
