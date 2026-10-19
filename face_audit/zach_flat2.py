"""zach_flat2.py SEEDS: flat emblem pasted so its 'Fortune' and 'Hunters' baselines land on the user's two red guide lines (rotation, scale, position), old badge blanked; klein wraps it. env TAG, SCALE_ADJ"""
import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit"); sys.path.insert(0, "/home/deck/workspace/csvm-hd/tools")
import numpy as np, cv2
from PIL import Image
import zach_guides as G, tile_upscale as tu
Z = "/home/deck/workspace/csvm-hd/face_audit/zach/"; TAG = os.environ.get("TAG", "g31"); ADJ = float(os.environ.get("SCALE_ADJ", "1.0"))
A = [G.window(p) for p in G.lines_in_anchor()]                                   # guide polylines in klein-window px (1024)
g1m, g2m = A[0][1:3].mean(0), A[1][:2].mean(0)
# baselines of the flat texture (HD px): Fortune from F foot to e, Hunters from H foot to s  (640-view coords x1.6)
f0, f1 = np.array([110, 452]) * 1.6, np.array([470, 458]) * 1.6; h0, h1 = np.array([205, 585]) * 1.6, np.array([545, 530]) * 1.6
# similarity transform: texture-line direction (Fortune) -> guide-1 direction, gap -> gap
ang = np.arctan2(*(A[0][-1] - A[0][0])[::-1]) - np.arctan2(*(f1 - f0)[::-1])
cross2 = lambda a, b: a[0] * b[1] - a[1] * b[0]
gap_tex = abs(cross2((f1 - f0) / np.linalg.norm(f1 - f0), (h0 + h1) / 2 - (f0 + f1) / 2)); gap_win = abs(cross2((A[0][-1] - A[0][0]) / np.linalg.norm(A[0][-1] - A[0][0]), g2m - g1m))
s = gap_win / gap_tex * ADJ; print("rotation %.1f deg, scale %.3f, gap tex %.0f win %.0f" % (np.degrees(ang), s, gap_tex, gap_win))
Rm = s * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
t = g1m - Rm @ ((f0 + f1) / 2)
M = np.hstack([Rm, t[:, None]]).astype(np.float32)
em = np.asarray(Image.open(Z + "gypsy_hd.png").convert("RGBA"), np.float32) / 255; em[..., :3] = em[..., :3].mean(-1, keepdims=True)
wp = cv2.warpAffine(em, M, (1024, 1024), flags=cv2.INTER_AREA, borderValue=(0, 0, 0, 0))
ref = np.asarray(Image.open(Z + "ref.png").convert("RGB"), np.float32)
Cc = (298, 448); old = np.zeros((1024, 1024), np.float32); cv2.ellipse(old, Cc, (345, 178), 62, 0, 360, 1.0, -1)
old = cv2.GaussianBlur(old, (0, 0), 6)[..., None]; plate = ref * (1 - old) + ref.mean(-1, keepdims=True) * 0.30 * old
a = wp[..., 3:]; out = plate * (1 - a) + wp[..., :3] * 255 * a
Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(Z + f"{TAG}_in.png")
P = ("In this black and white photo of an aircraft nose, a flat emblem (a crystal ball with a woman holding tarot cards and the script 'Fortune Hunters') has been laid over the fuselage. "
     "Paint that emblem as a decal applied to the curved fuselage: it must wrap around the curved metal like it is printed on a cylinder, with strong perspective foreshortening, "
     "so the crystal ball becomes a narrow slanted oval that follows the surface and curves away toward the upper right where the fuselage turns. "
     "Keep the position, size and the slant of the two lettering lines exactly as laid out: 'Fortune' on the first line and 'Hunters' on the second line, both running down toward the lower right at the same angle. "
     "Keep the artwork and lettering of the emblem the same, only warped onto the surface with matching light and film grain, and write the title only once. "
     "Remove everything of the old badge. Keep the sky, the gun barrel, the hand and the sleeve exactly as they are.")
for sd in [int(x) for x in sys.argv[1:]]:
    tu.locked_gen(Z + f"{TAG}_in.png", Z + f"{TAG}_s{sd}.png", P, 1024, sd); print("seed", sd, flush=True)
