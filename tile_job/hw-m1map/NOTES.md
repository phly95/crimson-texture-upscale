# hw-m1map notes (tiled FLUX 4x, per-tile prompts)
Final prompts: tile_job/hw-m1map/prompts.json (s2_00..11, s4_00..23; prompts_s4.json from the earlier agent is not used). Assembly: seam_assemble.py + seams_s2.json / seams_s4.json (copied from ha-m2map: ramps placed in text-free rows/columns).
Map: "Dreaming of becoming an AVIATOR" legend (compass, kilometers/miles bars, banner, Copies + two paragraphs, zeppelin caption, STRATEGIC logo, AIR FREIGHT COMPANY, HOLLYWOOD CALIFORNIA, STRIKE FORCE 48 box) and a town-and-bay map. Map numbers: 136 (sea, left), 450 (sea, centre and bottom right, cut by frame), 136 and 450 again in the s4 bottom tiles.

## Lessons
- Legend lines: transcribe each line verbatim with its spelling ('Pirates Society , Seattle Washington 300184.', 'in a process that is only done bye Atlas Paper'). The model keeps them.
- Tiles where a legend paragraph is cut by the tile top edge (s4_20): if the prompt omits the cut lines the model blurs them. Listing the two cut lines ('in a process ... Atlas Paper', 'Company.') fixed it.
- A negative sentence about a streak ('no streak') still drew a streak; a long bright diagonal bar appeared in s4_03 attempt 2. The clean result came from plain wording only ('the upper right land is plain soft khaki colour').
- Town blocks in the reference are blurred. Prompts that say 'blurred town texture, no separate drawn houses or trees' keep them blurred; without it the model draws roads and individual houses on the khaki land (s4_13 attempt 1 rejected; fixed with 'no roads, no buildings' on the upper right land).
- Invented boxes: s4_02 drew square boxes in the middle land; 'plain soft colour with no shapes on it' fixed it.
- A grey ink blot (bottom left of the legend) and the ring stain (bottom right of the sea) are part of the reference; the model keeps them.

## Per tile
- s2_00: kept from the earlier run (faithful legend).
- s2_01: redraw 1 (seed 457): the earlier version invented an airstrip with printed-looking marks and a block of buildings; blurred town only.
- s2_10: kept from the earlier run (legend, compass, banner, logos, scale bars all present).
- s2_11: kept from the earlier run (136, 450 x2, ring stain, bay).
- s4_00: redraw 1 (seed 458), legend correct.
- s4_01: redraw 1 (seed 459), legend edge pieces intact.
- s4_02: 2 attempts; the first drew boxes in the middle land, fixed (seed 461).
- s4_03: 3 attempts; attempt 1 speckle patch, attempt 2 long bright streak, attempt 3 (seed 463) clean.
- s4_10: redraw 1 (seed 459): all paragraphs correct, the Atlas and binded spellings kept.
- s4_11: redraw 1 (seed 460): town, grey complex, bay and cut 136 correct.
- s4_12: redraw 1 (seed 461): harbour, pier, buildings correct.
- s4_13: 2 attempts; attempt 1 invented roads and houses on the upper right land; attempt 2 (seed 463) clean.
- s4_20: 2 attempts; attempt 1 blurred the cut top paragraph lines (fixed by listing them, seed 461).
- s4_21: redraw 1 (seed 461): 136, 450, pier, box edge correct.
- s4_22: redraw 1 (seed 462): both 450 numbers, islets and channel correct; the town is drawn sharper than the blurred reference, same place.
- s4_23: redraw 1 (seed 463): 450 cut by the frame, ring stain, coast correct; the town is sharper, same place.

Still imperfect: tiny legend letters, e.g. 'Washingtоn' and the caption 'Cruise the world on a luxury liner' (reads soft and garbled at 4x); 'derivea' and 'Attas' in the paragraphs. The infill step should fix these.
