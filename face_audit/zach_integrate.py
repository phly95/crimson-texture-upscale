"""zach_integrate.py SEEDS: klein pass that paints the composite (card logo + script) as aircraft nose art"""
import sys; sys.path.insert(0, "/home/deck/workspace/csvm-hd/tools")
import tile_upscale as tu
P = ("Make this look like a real photograph of hand-painted nose art on the side of an aircraft fuselage, in black and white. The painted artwork is a fan of four tarot-style cards inside a dark oval "
     "badge: from the upper left to the lower right the cards read 'JUSTICE' (a propeller airplane over scales), 'WEALTH' (a big coin with a dollar sign), 'LOVERS' (a woman's portrait) and 'DEATH' (a skull). "
     "Keep every card, its title and its picture exactly as in the reference, only paint it as matte paint on riveted metal with slight wear. Over the cards runs the white brush-script painted title: "
     "'Fortune' on the first line and 'Hunters' on the second line, exactly seven letters each (F o r t u n e / H u n t e r s), thin clean flowing script. "
     "Keep the sky, the gun barrel, the rivets, the panel lines and the hand and sleeve at the right exactly as in the reference. Do not add any other lettering or any pirate skull with a hat.")
for s in [int(a) for a in sys.argv[1:]]:
    tu.locked_gen("face_audit/zach/comp.png", f"face_audit/zach/int_s{s}.png", P, 1024, s)
    print("seed", s, flush=True)
