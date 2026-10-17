"""cmp_sheet.py DIR OUT_PREFIX [N per sheet]: rows = stem | input | Qwen (as the user saw it) | Gemini, same klein seed."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw
d = Path(sys.argv[1]); pre = sys.argv[2]; per = int(sys.argv[3]) if len(sys.argv) > 3 else 6
S = 340
rows = [p for p in sorted(d.iterdir()) if (p / "gemini_s1.png").exists() and (p / "qwen_s1.png").exists()]
for k in range(0, len(rows), per):
    chunk = rows[k:k + per]
    im = Image.new("RGB", (3 * S + 10, len(chunk) * (S + 22)), (30, 30, 30))
    dr = ImageDraw.Draw(im)
    for i, p in enumerate(chunk):
        y = i * (S + 22)
        dr.text((4, y + 4), f"{p.name}      input | Qwen | Gemini", fill=(255, 255, 255))
        for j, f in enumerate(("in.png", "qwen_s1.png", "gemini_s1.png")):
            t = Image.open(p / f).convert("RGB"); t.thumbnail((S, S))
            im.paste(t, (j * (S + 5), y + 20))
    im.save(f"{pre}{k // per + 1}.png")
    print(f"{pre}{k // per + 1}.png", [p.name for p in chunk])
