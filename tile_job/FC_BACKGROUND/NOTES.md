# FC_BACKGROUND notes (tiled FLUX 4x, per-tile prompts)
Final prompts: tile_job/FC_BACKGROUND/prompts.json. Assembly: seam_assemble.py + seams_s2.json / seams_s4.json (wide ramps on blank paper areas, narrow ones next to text).
Lettering in the picture: brass bar 'Clip-O-Rama' (script, engraved), white boxes 'Aviation Form FC-1' and '96-7834-2a' at the top right. Everything else blank.

## Lessons
- Spelling the header strings digit by digit ('nine six, hyphen, seven eight three four, hyphen, two a') gives exact text. 'Aviatio' is a cut word at the s4_02 right edge: say so.
- The parchment keeps its soft look when the prompt says 'soft, smooth, blurry-looking ... draw no creases, wrinkles, crumples, fold lines or sharp ridges' (first wording 'faint creases' still invented sharp ones).
- Do not write 'gold hairline scratches' in a prompt that also shows parchment: s4_13 drew gold scratch lines across the note. Give a board-only scratch sentence, and say 'completely free of lines' for the note tile.
- A hint that says 'wooden board on dark red wood table' made pure-paper tiles (s4_11, s4_12) grow pink/teal wood at their edges; use a neutral hint and say 'the entire tile edge to edge is paper, no wood, no border'.
- s2_11 once grew a transparent plastic clip and a 2x2 grid of boxes in the sheet (the ref has none): say 'exactly two blank fields, nothing else on the sheet, no grid, no cells'.

## Per tile
- s2_00, s2_01, s2_10: first prompt accepted (Clip-O-Rama reads 'Clip-O-Rama' with last letters 'Rama' softly; header text exact).
- s2_11: 3 attempts (clip + grid invented, then clean).
- s4_00, s4_01, s4_02, s4_03, s4_10, s4_20, s4_21, s4_22, s4_23: accepted at first draw. s4_03 header text exact; s4_10 'Clip-O-Rama' present (the final letters read 'Rama' like the ref).
- s4_11, s4_12: 2 attempts (wood at edges, fixed by neutral hint + edge-to-edge paper wording).
- s4_13: 2 attempts (gold scratches over parchment, fixed; seed 7).
Still imperfect: tiny step (~3 px at 4x) in the long field's navy lines where tile columns meet near x=1600; board scratches in s4_00 are slightly stronger than the ref; s4_01 field edge line has a slight kink near the right.
