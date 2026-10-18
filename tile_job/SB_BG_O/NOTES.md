# SB_BG_O notes (tiled FLUX 4x, per-tile prompts)
Prompts: prompts.json. Assembly: tools/seam_assemble2.py + seams_s2.json / seams_s4.json (x seams in ink-free columns between masthead letters: 1013/1589/2201 ramp 16; y 870/1544 ramp 100).
Lettering: 'EMPIRE STATE EXAMINER' serif masthead, only the bottom ~25px (2x) cut by the top edge, then 2 rules, double zigzag, rule.
## Lessons
- Cut-off masthead: saying "cut off by the top edge, only lower part shows" made s2_00 redraw complete letters / a whole new masthead below. What worked: describe a thin band ("a band of flat dark grey ink only about 25 (55 at s4) pixels tall: the bottom strip of a line of serif capitals whose upper parts lie outside the crop") + quote the visible letters with the cut-edge letters named.
- s4_23 invented a second torn paper edge under the real one; "one single ragged torn edge, smooth flat grey-green surface with soft pale streaks" + seed 71 fixed it.
## Per tile
- s2_00: 4 attempts (full letters, new masthead, then band wording ok). s2_01: 2. s2_10/11: 1. s4: 1 attempt each except s4_23 (2).
Imperfect: zigzag line phase differs slightly between tiles at the seam columns (hardly visible).
