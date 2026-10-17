"""sheet.py OUT.png NAME [extra out files...]: original | Nano Banana | each Qwen output, on a checkerboard, 1024 wide each scaled to S."""
import sys
from PIL import Image, ImageDraw
import numpy as np
out, name, *runs = sys.argv[1:]
S = 400
def chk(n):
    g = (np.indices((n, n)).sum(0) // 16) % 2
    return Image.fromarray(np.where(g[..., None], [60, 60, 66, 255], [34, 34, 40, 255]).astype("uint8"), "RGBA")
def fit(p, nearest=False):
    im = Image.open(p).convert("RGBA")
    k = min(S / im.width, S / im.height); im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.NEAREST if nearest else Image.LANCZOS)
    c = chk(S); c.alpha_composite(im, ((S - im.width) // 2, (S - im.height) // 2)); return c.convert("RGB")
tiles = [(f"in/{name}_orig.png", "original (nearest)", True), (f"nb/{name}.png", "Nano Banana 2.1", False)] + [(r, r.split("/")[-1][:34], False) for r in runs]
sheet = Image.new("RGB", (len(tiles) * (S + 6), S + 18), (20, 20, 24)); d = ImageDraw.Draw(sheet)
for i, (p, lab, nn) in enumerate(tiles):
    sheet.paste(fit(p, nn), (i * (S + 6), 0)); d.text((i * (S + 6) + 2, S + 3), lab, fill=(230, 230, 230))
sheet.save(out); print(out, sheet.size)
