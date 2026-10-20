"""zach_gint.py SEEDS: klein pass painting the crystal-ball composite as nose art on the fuselage"""
import sys; sys.path.insert(0, "/home/deck/workspace/csvm-hd/tools")
import tile_upscale as tu
P = ("Make this look like a real black and white photograph of an aircraft with a round-cornered oval nose-art emblem of a crystal ball on the metal fuselage: inside the glass crystal ball a woman fortune teller "
     "with dark wavy hair holds a fan of tarot cards, with soft smoke around her, in a chrome rim, and the script title 'Fortune Hunters' (two words, F-o-r-t-u-n-e and H-u-n-t-e-r-s) across the lower part. "
     "Keep the emblem's design, the woman, the cards and the lettering exactly as in the reference; only make it sit naturally on the curved riveted metal with the same lighting and film grain as the rest of the photo. "
     "Remove any leftover old badge edges, stray letters or marks around the emblem, restoring plain metal with rivets and panel lines. Keep the sky, the gun barrel, the hand and the sleeve at the right exactly as in the reference. Do not add other lettering.")
for s in [int(a) for a in sys.argv[1:]]:
    tu.locked_gen("face_audit/zach/gcomp.png", f"face_audit/zach/gint_s{s}.png", P, 1024, s)
    print("seed", s, flush=True)
