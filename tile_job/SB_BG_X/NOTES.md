# SB_BG_X notes (tiled FLUX 4x, per-tile prompts)
Prompts: prompts.json. Assembly: tools/seam_assemble2.py + seams_s2/s4.json (x seams at min-diff columns 767/1600/2228 ramp 16; y 880/1544 ramp 100; s2 x 716 ramp 12).
Lettering: logo 'ELECTRIC' / 'SERVICE' (+ cut letters 'A','E' by the top edge), 'HOLLYWOOD SIGNAL' on an arc, blue-ink signature scribble (no words).
## Lessons
- Cut letters at the top edge ('A', 'E' slivers beside the tower): naming them made s2_00 redraw two full big letters 'A' and 'E' with glow. Say "only tiny black slivers (bottom tips of letters) show, no full letters there".
- Describing 'the end of a long blue line' for s2_01 made it invent a faint streak; say 'one tiny blue dot at the left edge, no other handwriting'.
- Handwriting: 'a tangle of a few long scribble strokes ... keep layout, do not invent readable words' worked; strokes reproduced.
- Adding "printed flat in black ink, crisp, no shadow, no glow, no emboss" to s4 logo tiles kept lines flat; s2 had a mild emboss glow around black lines that is still faintly visible.
- Frame lines drawn by two neighbouring tiles differ by 1-2 px in y at x seams: use narrow ramps (12-16) at the column of minimal difference.
## Per tile
- s2_00: 2 attempts, s2_01: 3, s2_10/11: 1. s4: all 12 first draw.
Imperfect: faint dark halo along thick black lines.
