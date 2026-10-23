"""mag_compose.py COVER [OUT.png]: stack a cover's parts (pipeline/mag_parts/COVER/manifest.json) in z order, apply the original page alpha -> OUT (default pipeline/mag_parts/COVER/composed.png)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mag_parts as mp

if __name__ == "__main__":
    cover = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else str(mp.PARTS / cover / "composed.png")
    im = mp.compose(cover, out)
    print("wrote", out, im.size)
