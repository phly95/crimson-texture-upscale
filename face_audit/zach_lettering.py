"""zach_lettering.py SEEDS: redraw the whole 'Fortune Hunters' lettering of the anchor emblem in one enlarged window (letters ~2x bigger for the model)."""
import sys; sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit"); sys.path.insert(0, "/home/deck/workspace/csvm-hd/tools")
from PIL import Image
import tile_upscale as tu
LB = (0, 190, 540, 730)
Image.open("face_audit/zach/ref.png").convert("RGB").crop(LB).resize((1024, 1024), Image.LANCZOS).save("face_audit/zach/let_ref.png")
P = ("Redraw this crop of a skull-and-pirate-hat emblem painted on a dark aircraft panel, with its white hand-painted brush-script lettering, crisp and clean. The lettering is two words: "
     "'Fortune' on the upper line and 'Hunters' on the lower line, both in the same flowing script, running diagonally down to the right. 'Fortune' has exactly seven letters: F, o, r, t, u, n, e. "
     "'Hunters' has exactly seven letters: capital H (two vertical stems joined by a crossbar), then u, n, t, e, r, s. Keep the skull, hat, swords, badge outline, positions, slant, size, "
     "colours, metal and rivets exactly as in the reference. Do not add any other letters, dots, marks or symbols.")
for s in [int(a) for a in sys.argv[1:]]:
    tu.locked_gen("face_audit/zach/let_ref.png", f"face_audit/zach/let_s{s}.png", P, 1024, s)
    print("seed", s, flush=True)
