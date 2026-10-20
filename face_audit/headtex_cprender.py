"""headtex_cprender.py TEX.png OUT.png [model pack]: 4-view mesh render sheet (front, 3/4, side, back) of a pilot head texture, as the engine would sample it (flipv False, fixuv True = collapsed right-goggle UVs borrowed from the mirror triangles, i.e. the corrected preview; the game still shows the original streak)"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_render as HR
def sheet(tex, S=480, fixuv=True, views=((180, 0), (135, 15), (90, 0), (0, 0))):
    return np.hstack([HR.render({"cphead": tex}, y, p, S=S, flipv=False, model=70, pack="planes.zip", fixuv=fixuv, engine=True) for y, p in views])
if __name__ == "__main__":
    t = np.array(Image.open(sys.argv[1]).convert("RGB")); Image.fromarray(sheet(t)).save(sys.argv[2])
