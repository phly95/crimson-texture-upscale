# SB_19_01_DRINK notes (tiled FLUX 4x, per-tile prompts)
Prompts: prompts.json (all recipe lines quoted incl. cut fragments). Assembly: seam_assemble.py (= tools/seam_assemble2.py) + seams_s2/s4.json (min-diff columns/rows, ramp 16). Final submitted = tile_job/SB_19_01_DRINK_4x_text.png (unpolished tiles, first-draw per tile).
## Lessons
- Interior text tiles: describing "a card with a teal frame line and wavy border" makes the model draw that frame/edge in tiles that only show inside paper (s4_01, s4_11, s4_12 got invented frames + ghost duplicate text). Fix: "inside of a white-grey paper recipe card ... whole crop is plain paper edge to edge: no frame line, no border, no card edge, no page beyond it".
- Adding 'printed flat, no shadow, no glow, no emboss' removed the embossed look of the lettering.
- polish_text_tiles.py (best-of-6 seeds + word stitching) pasted a ghost duplicate '--Swift Jimmy' into the signature area and had been auto-submitted; I restored the first-draw tiles (polished_bak/ holds the polished ones), reassembled and re-submitted. Always view the polished result before trusting the auto submit.
## Per tile
- s2: 2 attempts (emboss glow), s4: 1 attempt each except s4_01/11/12 (2).
Imperfect: card top-left corner shows a small translucent label at top edge only roughly drawn.
