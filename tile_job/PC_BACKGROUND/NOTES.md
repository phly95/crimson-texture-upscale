# PC_BACKGROUND notes (redone from scratch; first agent's result discarded -> tile_job/_old_PC_BACKGROUND_agent1)
Old result was bad: model filled the pure-black window with invented machinery/desk, duplicated blueprint pens/compass, seams.
Fixes: (1) tools/blackfix.py NAME in out: source pixels exactly 0 (large connected areas = window + frame interior) forced back to pure black after assembly (s2 stage too, so s4 refs stay black). (2) prompts describe ONLY what is visible in each crop.
Lettering: 'R' on radio (s4_10, s4_20), 'Aloha Daily' masthead (s4_22), blueprint lines 'Name: Grumman EWC?', 'Weight Capacity:', '2100 lbs', 'Overall Speed: High', 'Structure Integrity: Poor' (s4_23; the original itself is near-illegible handwriting).
## Lessons
- Describing an object that is only partly in the crop makes klein draw the whole object inside the crop (s4_20 drew a complete radio; s4_12 grew an inkwell+blueprint+desk and shifted the wall). Say 'the lower half of X is cut by the top edge', 'only along the bottom edge ... cut by the bottom edge, nothing else in the crop'.
- Words like 'compass' / 'compasses' in a tile prompt invent extra compasses; use 'exactly one ... exactly one pair of dividers'.
- s4_23: garbled hand lettering in the s2 ref is reproduced, not repaired; seeds only vary the blotch near 'High'/'Poor'. Took the first (a1) draw's text block and blended it into the later draw with the correct radio (a1 had invented a second radio above the real one).
## Per tile
s2: 1 draw each (00,01,10,11). s4: 00-03,10,11,13,22 first draw; s4_12 2 draws, s4_20 2, s4_21 2, s4_23 3 + 4 seeds + manual text-block blend.
Seams: seams_s4.json (seam_assemble2.py).
