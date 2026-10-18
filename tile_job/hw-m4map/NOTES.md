# hw-m4map notes (tiled FLUX 4x, per-tile prompts)
Final prompts: tile_job/hw-m4map/prompts.json (s2_00..11, s4_00..23). Assembly: seam_assemble.py + seams_s2.json / seams_s4.json (copied from ha-m2map).
Map: the same "Dreaming of becoming an AVIATOR" legend as hw-m1map, with a blue ruled sea (dial at upper left, KE/JE/KG/JG labels, green numbers 450, 136, 311, 1466, two long solid lines, one dashed curve).

## Lessons
- Dial: the model drew an extra dial in the right-middle sea (s2_01). Saying 'exactly one dial, at the upper left, no other dial, compass, arrow or circle' stopped it.
- The labels KE/JE/KG/JG are sharp and unchanged; the small numbers keep their place when the prompt gives each one's position relative to a ruler line.
- In the bottom-left sea (s4_21) the first draw invented a pale pointed object with a blue outline across the sea. Calling the sea 'plain mottled blue water, empty and uniform, with no shapes' fixed it.
- The legend column can reuse hw-m1map's accepted legend wording; only the sea-strip sentence changes (ruler ticks and frame instead of khaki land).

## Per tile
- s2_00: kept from the earlier run (legend and sea numbers verified).
- s2_01: redraw 1 (seed 457): extra dial removed, KE/JE/KG/JG and 450, 136, 311 in place.
- s2_10: kept from the earlier run.
- s2_11: 1 redraw (seed 458): labels, cut dial, 136, 450, 311, 1466 and the ring stain correct.
- s4_00: redraw 1 (seed 458), legend correct.
- s4_01: redraw 1 (seed 459), dial, KE and the dashed curve correct.
- s4_02: redraw 1 (seed 460), cut KE/JE correct.
- s4_03: redraw 1 (seed 461), JE and the solid line correct.
- s4_10: redraw 1 (seed 459), all paragraphs correct.
- s4_11: redraw 1 (seed 460), KG and the dashed curve correct.
- s4_12: redraw 1 (seed 461), cut KG and JG correct.
- s4_13: redraw 1 (seed 462), 136 and cut 311 correct.
- s4_20: redraw 1 (seed 460), cut top lines, zeppelin, STRATEGIC logo and STRIKE FORCE box correct.
- s4_21: 2 attempts; attempt 1 drew a pale object across the sea (seed 461), attempt 2 (seed 462) clean.
- s4_22: redraw 1 (seed 462), cut KG, JG, 450 correct.
- s4_23: redraw 1 (seed 463), JG, 311, 1466 and the ring stain correct.

Still imperfect: the 311 under the ruler reads softly (partly hidden in the reference too); the JG pair in s4_12 is a little sharper than the blurred reference.
