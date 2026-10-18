"""Recipe for abld_shadow: building shadow flat alpha fill layer
with exact constant alpha (125/255 = ~0.4902) and pure black RGB.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import fx

def make_abld_shadow():
    W, H = 512, 512
    orig = fx.load("sprites/abld_shadow/orig.png")
    a_val = orig[..., 3].mean()

    # Perfectly flat constant alpha layer matching original exactly
    a_hd = np.full((H, W), a_val, dtype=np.float32)
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    layer = np.dstack([rgb, a_hd])

    out_dir = Path("sprites/abld_shadow")
    fx.save(layer, out_dir / "draw.png")
    print("Saved sprites/abld_shadow/draw.png")

if __name__ == "__main__":
    make_abld_shadow()
