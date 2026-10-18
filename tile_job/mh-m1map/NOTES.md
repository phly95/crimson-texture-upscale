# mh-m1map notes (tiled FLUX 4x, per-tile prompts)
Final prompts: tile_job/mh-m1map/prompts.json (s2_00..11, s4_00..23). Assembly: seam_assemble.py + seams_s2.json / seams_s4.json (copied from ha-m2map).
Map: "New York" (PUBLISHING COMPANY / DECO PRINTING legend, a brown kraft column, the Copies paragraphs, two badges, compass, scale bars) with a dense charcoal street map. Labels read from crops: NEW YORK, Murray Hill, Chelsea, Manhattan, Gramercy Park, West Village, Greenwich Village, SoHo, Little Italy, Tribeca, Chinatown, Lower East Side, Stuyvesant Town, SEA, Que, Williamsburg, South Street Seaport, cut 'lyn'/'klyn'.

## Lessons
- The old s2_01/s2_11 prompts said 'no large lettering' and 'the map holds no lettering'. Both were wrong for this map: it has many medium labels. Read every label from crops before writing the prompt.
- Labels spelled out one by one in the prompt (e.g. 'WEST' above 'VILLAGE') placed them in the right spots.
- A label appears once: the model repeated 'LITTLE ITALY' (s2_11 attempt 1) and drew 'CHELSEA' in a box and then in large bold type (s2_10 attempts 2-3). Say 'exactly once', 'no box', 'the same small size as Murray Hill' and state the position.
- A word in the legend lines can be lost or garbled ('Copies' paragraph got a duplicated line in s2_10 attempt 3); it cleared on a fresh draw with the same wording.
- Invented shapes came from copying a sentence across tiles: a 'brown stain blot' sentence from s4_21 put a large stain in s4_23 (attempt 1). Check each sentence against that tile's own reference.
- In s4_00 the model drew cyan water in the map strip next to the legend (attempt 1). 'Contains no cyan water and no blue at all' fixed it.

## Per tile
- s2_00: redraw 1 (seed 456): legend, NEW YORK, Murray Hill, cut 'Manhat' correct.
- s2_01: redraw 1 (seed 457): top labels (Chelsea, Manhattan, Gramercy Park, Stuyvesant Town, Que, SEA, Murray Hill) correct.
- s2_10: 4 attempts; attempt 1 'Queens'-like word at the top right (seed 457); attempt 2 CHELSEA in large bold (seed 458); attempt 3 boxed label and duplicated Copies line (seed 459); attempt 4 (seed 457) clean: CHELSEA removed from this tile, the top right is plain street.
- s2_11: 2 attempts; attempt 1 a second LITTLE ITALY and a bold 'lyn' (seed 458); attempt 2 (seed 459) clean.
- s4_00: 2 attempts; attempt 1 cyan water in the map strip and 'NE' (seed 458); attempt 2 (seed 459) clean.
- s4_01: redraw 1 (seed 459): NEW YORK, Murray Hill, cut 'W / VIL' correct.
- s4_02: 2 attempts; attempt 1 CHELSEA in a dark box at the top left (seed 460); attempt 2 (seed 461) plain label at the left third.
- s4_03: redraw 1 (seed 461): Que, SEA, Stuyvesant Town, cut Gramercy Park correct.
- s4_10: redraw 1 (seed 459): legend paragraphs, Copies lines, badges correct.
- s4_11: 2 attempts; attempt 1 lost the cut legend fragments d / ir / er / xy / nal (seed 460); attempt 2 (seed 461) kept them, Tribeca and the cut W / VIL correct.
- s4_12: redraw 1 (seed 461): Gramercy Park, West Village, Greenwich Village, SoHo, Little Italy, Tribeca, Chinatown, Lower East Side correct.
- s4_13: redraw 1 (seed 462): Stuyvesant Town, SEA, East Village, Williamsburg, Lower East Side correct.
- s4_20: redraw 1 (seed 460): Copies lines, badges, compass, scale bars correct.
- s4_21: redraw 1 (seed 461): stamp, cut 'xy' and 'nal', cut SO/SE correct.
- s4_22: redraw 1 (seed 462): Chinatown, Lower East Side, South Street Seaport, cut 'lyn' correct.
- s4_23: 2 attempts; attempt 1 invented a large brown stain at the lower left (seed 463); attempt 2 (seed 464) clean.

Still imperfect: tiny paragraph letters (e.g. 'Atlas Paper', 'Seeatle Washhington' in the Copies line) read soft; the infill step should fix them. The s2_11 top-left still says 'Murray Italy' in the discarded strip and does not reach the composite.
