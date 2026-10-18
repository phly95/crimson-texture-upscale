# ha-m1map notes (tiled FLUX 4x, per-tile prompts)
Final prompts: tile_job/ha-m1map/prompts.json (s2_00..11, s4_00..23). Assembly: seam_assemble.py + seams_s2.json / seams_s4.json (hard-ish seams placed in text-free rows/cols, instead of the 256px feather which ghosts text that two tiles draw differently).

## Lessons
- Transcribe the small paragraphs: the model gets them right when given each line verbatim (the 'fir' / 'binded' / 'bye Atlas Paper' typos are in the original and are kept). Wording like "keep as is" without text gives garbled letters.
- Say the year digit by digit ("1931 with a 3, never 1921"); it then stays 1931.
- Cut-off text at tile edges: list the fragments ('roduced', 'om the fir', ...). Worked.
- The global hint must NOT mention things (legend column, "teal sea with contour lines") the tile does not contain: it made s4 tiles invent legend lists/tables in empty sea corners. Use a neutral hint ("detail crop ... never add anything not in the crop") and per-tile prompts that say "no lettering at all".
- "no legend box / table / list" in NOEXTRA text actually primed the model to draw them; say instead "The rest of the tile is sea, with no boxes" and "no other lettering".
- Legend tiles: wording that the grey column has a straight right edge and "nothing from the sea crosses it" reduced contours bleeding into the grey paper; the s2 tiles needed seed changes too.
- Compass: the model likes to put an N/W/R letter inside the compass wheel; telling it "only spokes and eight tiny arrow-heads" helps but small glyph-like marks remain (s4_10, s4_20).
- Each tile may fail with 'generate_image returned no results' when VRAM is contended by another job; just retry after ~20 s (wrapper script g.sh loops).

## Per tile
- s2_00: 5 attempts (generic prompt first, then verbatim text; seeds default, 777, 4242, 99). Issues: '1931'->'1921', compass got extra 'N' inside, header paragraph garbled. Fixed by verbatim transcription and digit-by-digit year; kept last (seed 99).
- s2_01: 2 attempts. First invented a text box/table in sea corners; added 'no legend box, table, text block, caption' -> clean.
- s2_10: 2 attempts (seed 31337 second). Text correct incl. 'kilometers/miles', HOLLYWOOD CALIFORNIA.
- s2_11: 2 attempts, clean; Copyright line correct.
- s4_00: seeds default, 13, 59 (final 59). Final: all header text and the three paragraphs correct except 'Colorado' -> 'Coloraido' slightly, and 'hea vy' kerning; 'GLOBAL' logo ok.
- s4_01: 5 attempts. Early ones invented a large light blob/shape in open sea and a legend table at tile left; fixed with neutral hint + 'open plain water, no new shapes'. Final seed 61 (still a faint light diagonal streak that is in the ref and a pale shape over the ocean contour; acceptable).
- s4_02, s4_03, s4_12, s4_21: 2 attempts (hint fix). All numbers (339/129/163/37/38) and 'Printed in December of 1931' correct.
- s4_10: 3 attempts; final has small paragraph words slightly wrong ('Holywood Calfonia' in the 2nd paragraph, 'Attas'). Compass has small glyph-like marks.
- s4_11, s4_13, s4_22, s4_23: first runs with 'legend' hint invented a legend table at the left edge; neutral hint + 'no boxes' fixed (s4_13 seed 41, s4_22 seed 42, s4_23 seed 43).
- s4_20: 5 attempts; one run ghosted the previous compass/REGIONAL logo (double image); final seed 57 clean: REGIONAL logo, DISTRIBUTING COMPANY, HOLLYWOOD, CALIFORNIA (last letter A looks like 'ìA'), kilometers, miles, two bars.
Still imperfect: tiny paragraph letters in s4_00/s4_10 (a few letters wrong, e.g. 'Holywood Calfonia', 'purchnaed', 'Atlas'->'Attas'); compass mini marks in s4_10/s4_20.
