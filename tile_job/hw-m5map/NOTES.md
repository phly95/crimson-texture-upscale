# hw-m5map notes (tiled FLUX 4x, per-tile prompts)
Final prompts: tile_job/hw-m5map/prompts.json (s2_00..11, s4_00..23). Assembly: seam_assemble.py + seams_s2.json / seams_s4.json (copied from ha-m2map).
Map: the same "Dreaming of becoming an AVIATOR" legend as hw-m1map, and a map of olive-green and khaki mottled ground with a thick black frame, a grid of thin dark lines, dashed and solid, bush patches, a soft fold band across the middle and a brown ring stain at the bottom right. There is no lettering on the map side.

## Lessons
- Negative sentences ('no cracks', 'no creases') alone do not stop the model from drawing cracks. Describing the fold as a soft blurred band with no sharp line worked (s4_22 attempt 4, s4_23 attempt 3).
- Frame wording: 'all of it map ground edge to edge, no frame, no border, no paper' stopped a paper border from being drawn (s4_12 attempt 1 drew a bright border and blue-black grid).
- Tree sprites: 'soft blurred colour areas only, no individual trees, no tree crowns' stopped tree-like sprites appearing (s4_13 attempt 3). Bush patches were then too big; 'no bigger than in the reference' was added.
- Crisp crowns vs blurred bushes: 'blurred edges with no crisp outlines and no tree-like crowns' fixed outlined crowns (s4_22 attempt 2).
- Box letters at the left edge of the legend strip: explicit 'the white digit 8 and the white letter E ... fully legible and sharp' fixed a blurred dark column (s4_21 attempt 1).
- Ring stains drawn too faint in attempt 1 (s4_23); 'drawn as strongly and as dark brown as in the reference' fixed it.

## Per tile
- s2_00: redraw 1 (seed 456), legend and banner correct; the caption is soft tiny print (accepted).
- s2_01: redraw 1 (seed 457), map ground only.
- s2_10: redraw 1 (seed 457), legend and the STRIKE FORCE box correct.
- s2_11: redraw 1 (seed 458), ground, grid and ring stain correct.
- s4_00: redraw 1 (seed 458), legend correct.
- s4_01: 2 attempts; attempt 1 drew a hard-edged khaki shape in the bottom right (seed 459), attempt 2 (seed 461) a soft gradient.
- s4_02: redraw 1 (seed 460), ground correct.
- s4_03: 3 attempts; attempt 1 hairline cracks, attempt 2 cracks again (seed 462), attempt 3 (seed 463) clean.
- s4_10: redraw 1 (seed 459), all legend lines correct.
- s4_11: redraw 1 (seed 460), bushes and grid correct.
- s4_12: 2 attempts; attempt 1 paper border and blue-black grid (seed 461), attempt 2 (seed 462) clean.
- s4_13: 5 attempts; attempt 1 cracks (seed 463), attempt 2 bushes too big, attempt 3 tree sprites, attempt 4 crumples on the left edge, attempt 5 (seed 466) clean.
- s4_20: redraw 1 (seed 460), all legend items correct, the caption reads 'Cruise the world on a luxury liner'.
- s4_21: 2 attempts; attempt 1 garbled the box letters and invented paper creases (seed 461), attempt 2 (seed 462) clean.
- s4_22: 4 attempts; attempt 1 diagonal cracks (seed 462), attempt 2 outlined bush crowns (seed 463), attempt 3 crack-like fold (seed 464), attempt 4 (seed 465) clean.
- s4_23: 3 attempts; attempt 1 ring too faint (seed 463), attempt 2 crack-like fold (seed 464), attempt 3 (seed 465) clean.

Still imperfect: the tiny caption 'Cruise the world on a luxury liner' and some tiny paragraph letters are soft; the infill step should fix them. The faint fold line in the middle is a bit stronger than in the reference.
