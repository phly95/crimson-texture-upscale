"""zach_cards.py: composite of the Fortune Hunters card-fan logo (user-supplied) into the anchor's emblem oval, plus script lettering, as the reference for a klein integration pass."""
import sys, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
S = 1024 / 540
logo = Image.open("/home/deck/.claude/uploads/8d33977a-9b65-41a7-af0e-5ee75413ed5e/97abb752-image.webp").convert("RGBA")
ref = Image.open("face_audit/zach/ref.png").convert("RGB")
C = (298, 448); A, B, ANG = 335, 168, 62.0
if len(sys.argv) > 1 and sys.argv[1] == "ov":
    im = np.asarray(ref).copy(); cv2.ellipse(im, (int(C[0]), int(C[1])), (int(A), int(B)), ANG, 0, 360, (255, 0, 0), 3)
    Image.fromarray(im).save("face_audit/zach/ellipse_check.png"); sys.exit()
SC = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
# fan: grey, rotated so its x axis lies along the ellipse long axis
g = logo.convert("LA"); lw = int(2 * A * 0.98 * SC); lh = int(logo.height * lw / logo.width)
g = g.resize((lw, lh), Image.LANCZOS)
rot = g.rotate(-ANG, expand=True, resample=Image.BICUBIC)
base = ref.convert("L").convert("RGBA")
m = np.zeros((1024, 1024), np.uint8); cv2.ellipse(m, (int(C[0]), int(C[1])), (int(A * 0.97), int(B * 0.97)), ANG, 0, 360, 255, -1)
plate = np.asarray(ref.convert("L"), np.float32)
dark = cv2.GaussianBlur(np.where(m > 0, 38.0, 0).astype(np.float32), (0, 0), 3)
mm = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 3)
plate = plate * (1 - mm) + 38.0 * mm
canvas = Image.fromarray(plate.astype(np.uint8)).convert("RGBA")
layer = Image.new("LA", canvas.size, (0, 0)); layer.paste(rot, (int(C[0] - rot.width / 2), int(C[1] - rot.height / 2)), rot)
la = np.asarray(layer, np.float32); a = la[..., 1] / 255 * mm
canv = np.asarray(canvas.convert("L"), np.float32)
out = canv * (1 - a) + la[..., 0] * a
im = Image.fromarray(out.astype(np.uint8)).convert("RGB")
# script lettering: each word rendered flat, fitted to a length, rotated onto its own slope
import subprocess
FONT = subprocess.run(["fc-match", "-f", "%{file}", "Freestyle Script"], capture_output=True, text=True).stdout
def word(txt, length, angle, cen):
    f = ImageFont.truetype(FONT, 220); bb = f.getbbox(txt, stroke_width=3)
    t = Image.new("L", (bb[2] + 40, bb[3] + 40), 0); ImageDraw.Draw(t).text((20, 20), txt, font=f, fill=255, stroke_width=3, stroke_fill=255)
    t = t.crop(t.getbbox()); k = length / t.width; t = t.resize((int(t.width * k), int(t.height * k)), Image.LANCZOS)
    t = t.rotate(-angle, expand=True, resample=Image.BICUBIC)
    ta = np.zeros((1024, 1024), np.float32); x0, y0 = int(cen[0] - t.width / 2), int(cen[1] - t.height / 2)
    arr = np.asarray(t, np.float32) / 255
    ys, xs = max(0, y0), max(0, x0); ye, xe = min(1024, y0 + t.height), min(1024, x0 + t.width)
    ta[ys:ye, xs:xe] = arr[ys - y0:ye - y0, xs - x0:xe - x0]
    return ta
res = np.asarray(im, np.float32)
for txt, L, ang, cen in (("Fortune", 400, 40, (272, 470)), ("Hunters", 330, 25, (312, 650))):
    ta = word(txt, L, ang, cen)
    res = res * (1 - ta[..., None]) + 240 * ta[..., None]
Image.fromarray(res.clip(0, 255).astype(np.uint8)).save("face_audit/zach/comp.png")
