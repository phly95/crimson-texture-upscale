"""zach_word.py: redraw only the word 'Hunters' of the emblem lettering (crop enlarged so klein sees the letters big)."""
import sys; sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit"); sys.path.insert(0, "/home/deck/workspace/csvm-hd/tools")
from PIL import Image
import tile_upscale as tu
ref0 = Image.open("face_audit/zach/ref.png").convert("RGB")          # emblem crop of the anchor at 1024
WB = (100, 440, 520, 860)                                            # square window around the word
ref0.crop(WB).resize((1024, 1024), Image.LANCZOS).save("face_audit/zach/word_ref.png")
P = ("Redraw this crop of white hand-painted brush-script lettering on a dark aircraft panel, crisp and clean. The lettering is the single word 'Hunters' written in flowing "
     "script, exactly seven letters in this order: capital H, then lowercase u, n, t, e, r, s, running diagonally down to the right. The capital H has two vertical stems joined by "
     "a crossbar and nothing else. Keep position, slant, size, colour and the dark oval badge edge, the metal and rivets exactly as in the reference. Do not add any other letters, dots or marks.")
for s in [int(a) for a in sys.argv[1:]]:
    tu.locked_gen("face_audit/zach/word_ref.png", f"face_audit/zach/word_s{s}.png", P, 1024, s)
    print("seed", s, flush=True)
