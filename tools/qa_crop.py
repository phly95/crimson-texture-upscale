"""qa_crop.py NAME x0 y0 x1 y1 [OUT] [--cur FILE]: side-by-side of the original (Lanczos to the HD size) and the current HD page over grey, cropped to the box (HD pixel coordinates)."""
import sys
from pathlib import Path
from PIL import Image
G = Path("/tmp/claude-1000/-home-deck/8d33977a-9b65-41a7-af0e-5ee75413ed5e/scratchpad/gallery")
Q = Path("/tmp/claude-1000/-home-deck/8d33977a-9b65-41a7-af0e-5ee75413ed5e/scratchpad/qa")


def over(im):
    bg = Image.new("RGBA", im.size, (110, 110, 110, 255)); bg.alpha_composite(im); return bg.convert("RGB")


def main():
    a = sys.argv[1:]
    cur = None
    if "--cur" in a:
        i = a.index("--cur"); cur = a[i + 1]; del a[i:i + 2]
    n = a[0]; box = tuple(int(v) for v in a[1:5]); out = a[5] if len(a) > 5 else str(Q / f"{n}_crop.png")
    hd = Image.open(cur or G / f"{n}_hd.png").convert("RGBA")
    o = Image.open(G / f"{n}_orig.png").convert("RGBA").resize(hd.size, Image.LANCZOS)
    A, B = over(o).crop(box), over(hd).crop(box)
    w = Image.new("RGB", (A.width * 2 + 8, A.height), (0, 0, 0)); w.paste(A, (0, 0)); w.paste(B, (A.width + 8, 0)); w.save(out); print(out, w.size)


main()
