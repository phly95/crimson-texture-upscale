# nw-m3map notes (tiled FLUX 4x, per-tile prompts)

Final image: tile_job/nw-m3map_4x_text.png (3200x2400)
UID: rimage_nw-m3map__09009b0a
Assembly: seam_assemble.py + seams_s2.json / seams_s4.json

## Lessons
- s2 tiles:
  - s2_00, s2_01 kept from initial run.
  - s2_10 had an invented dial fragment at top right due to prompt phrasing; seam assembly at y=460 naturally eliminated it.
  - s2_11 generated and accepted cleanly.
- s4 tiles:
  - s4_00: FORRER TUNA COMPANY logo and paragraphs sharp on attempt 1.
  - s4_01: HE and 32° sharp on attempt 1.
  - s4_02: First prompt misidentified the grid crossing as JE; corrected to KE, rendered sharp with seed 460.
  - s4_03: Early attempt drew a large JE in open water; refined prompt to forbid open water text, JE on island rendered sharp.
  - s4_10: Scale bars had appeared above compass; clarified order and blank turquoise gap -> compass and scale bars sharp.
  - s4_11: Model mistook horizontal fold crease for a grid line and drew duplicated HG / 33° labels. Resolved by keeping HG/33° in row 2 (s4_21) and restricting s4_11 to HE/32° and island peak dial.
  - s4_12: KE at top and KG/450 at bottom sharp.
  - s4_13: JE at top, JG at bottom sharp on attempt 1.
  - s4_20: Compass rose, scale bars, CS badge, NEW YORK, NY sharp on attempt 1.
  - s4_21: Rendered HG, 33°, 136, 387 sharply below y=1550 seam.
  - s4_22: Brown island had grown a 450; restricted numbers to water below KG -> clean with seed 77.
  - s4_23: JG and bottom-right dial (0, 27, 6) sharp on attempt 1.
- Seam assemble:
  - s2 seams: x=800, y=[460, 600].
  - s4 seams: x=[835, 1500, 2240], y legend=[915, 1400], y map=[800, 1580]. Seam at x=835 lands precisely on the green frame line, preventing any text overlap between legend column and map.
- Remaining defects: minor typographic variations in tiny paragraphs, all map labels and dials match source exactly.
