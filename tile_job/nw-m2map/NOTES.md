# nw-m2map notes (tiled FLUX 4x, per-tile prompts)

Final image: tile_job/nw-m2map_4x_text.png (3200x2400)
UID: rimage_nw-m2map__d5d6870a
Assembly: seam_assemble.py + seams_s2.json / seams_s4.json

## Lessons
- s2 tiles:
  - s2_00, s2_01 kept from initial run.
  - s2_10 and s2_11 generated and accepted. Fixed number in bay from misread 367 to correct 387.
- s4 tiles:
  - s4_00: COLOSSAL logo, Paper and Timber Company, paragraphs clean on attempt 1.
  - s4_01: Clean on attempt 1.
  - s4_02: Attempt 1 grew a small 4-square yellow box near the circular facility; added negative wording forbidding yellow boxes and extra symbols, regenerated with seed 55 -> clean.
  - s4_03: Clean on attempt 1.
  - s4_10: Compass rose, scale bars, ATLAS, CS badge clean on attempt 1.
  - s4_11: 136 in bay clean on attempt 1.
  - s4_12: Runway and coast clean on attempt 1.
  - s4_13: Clean on attempt 1.
  - s4_20: NATIVE STREET emblem and publishing text clean on attempt 1.
  - s4_21: 136 and 387 clean on attempt 1.
  - s4_22: First runs with "upper middle" placed a duplicate 450 on the olive land. Refined prompt to explicitly specify the land has no numbers and 450 is only in the water on the upper right; regenerated -> clean.
  - s4_23: Clean on attempt 1.
- Seam assemble:
  - s2 seams: x=800, y=[450, 600].
  - s4 seams: x=[900, 1600, 2400], y legend=[895, 1550], y map=[950, 1550].
- Remaining defects: minor typographic variations in tiny paragraphs, but all lettering and numbers are crisp, accurate, and faithful to source.
