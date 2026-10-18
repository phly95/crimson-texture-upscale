# ha-m2map notes (tiled FLUX 4x, per-tile prompts)
Final prompts: tile_job/ha-m2map/prompts.json (s2_00..11, s4_00..23). Assembly: seam_assemble.py (copied from ha-m1map) with seams_s2.json / seams_s4.json (ramps only in text-free rows/cols).
Map: MAP OF HAWAII variant, same grey legend as ha-m1map (GLOBAL header, MAP OF HAWAII, GIANT, compass, REGIONAL, scale bars). Map numbers differ from ha-m1map: 846 (top edge), 163, 339 (x4), 129, 9 (right edge, cut by frame); copyright bottom right.

## Lessons
- Digit errors happen at tiny size: '646' drawn for '846' (fixed by spelling 'eight, four, six').
- A pond/island with small round dots: the model drew a letter-like glyph ('Z') in the pond; fixed with 'two tiny round dots, no letters, no digits and no glyphs'.
- Negative lists ('no boxes, no rectangles, no stripes') primed the model to draw boxes and stripes (s4_03, 3 attempts). A positive-only description ('the upper part of the sea is plain water with the faint lines of the reference') worked better.
- A number stated 'exactly once' (s4_01) stopped a duplicate 339 that was drawn in the open sea.
- A single tan bar in the island was drawn twice (crisp copy plus blurred ghost) in s4_13 attempt 1; and thin invented contour lines appeared on the island in attempt 2. Fixed by 'plain mottled green island, no line of any kind on it'.

## Per tile
- s2_00: kept from the previous run (verified clean).
- s2_01: redraw 1 (846 digits).
- s2_10: kept from the previous run (tiny 'Atlas' soft, acceptable).
- s2_11: redraw 1 (pond dots, no glyph).
- s4_00: redraw 1 (seed 458), legend correct.
- s4_01: redraw 2 (seed 461), duplicate 339 removed.
- s4_02: redraw 1 (seed 460), 846/163/129 correct.
- s4_03: 5 attempts; the last (seed 465) is clean: '3' at the left edge, 339, 'Printed in December of 1931'.
- s4_10: redraw 1 (seed 459), legend and compass correct.
- s4_11: redraw 1 (seed 460), 129 and red tip correct.
- s4_12: redraw 1 (seed 461), cut 129, pond dots, dock and tan bar correct.
- s4_13: 3 attempts; the last (seed 464) is clean.
- s4_20: redraw 1 (seed 460), REGIONAL block and scale bars correct.
- s4_21: redraw 1 (seed 461), clean.
- s4_22: redraw 1 (seed 462), 339 correct.
- s4_23: redraw 1 (seed 463), 339, 9 and copyright correct.

Still imperfect: tiny legend letters 'Secatte' for 'Seattle' (Copies line, s4_00/s4_10) and 'Attas' in the lower legend paragraph (s4_10); the infill step should fix these.
