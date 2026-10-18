# MM_SPLASHBACKGROUND notes (tiled FLUX 4x, per-tile prompts)
Final prompts: tile_job/MM_SPLASHBACKGROUND/prompts.json. Assembly: seam_assemble.py + seams_s2.json / seams_s4.json.
Lettering in the picture:
- Top banner: 'CRIMSON SKIES TM' (marbled red fill with silver bevelled frames)
- Top left badge: 'Microsoft', 'ZIPPER', 'INTERACTIVE' (white italic letters on black circle with pilot portrait)
- Airship hull: 'SON OF THE EMPIRE' (faint grey capitals along dark fuselage band behind propeller fighters)
- Tagline script: 'The Tall Tales and Many Lives of Nathan Zachary!' (thin light cream slanted cursive handwriting)

## Lessons
- In s2 and s4, the CRIMSON SKIES title letters are tightly spaced; placing x seams at exact bevel groove boundaries (base x=879 between R and I, x=1756 between CRIMSON and SKIES, x=2324 between E and S) eliminates bevel ghosting.
- The cursive tagline slopes upward across tile boundaries. Placed y seam in right zone at y=900 with narrow ramp (16px) between 'and' and 'Many Lives of', so lines 1-2 come cleanly from s4_03 and lines 3-4 come cleanly from s4_13 without crossing any letter stroke.
- For s4_13, describing only the lower tagline lines ('Nathan Zachary!') instead of mentioning lines 1-2 prevented the model from hallucinating shifted duplicate letters.

## Per tile
- s2: 4 tiles assembled cleanly with seam at x=878 and y=600.
- s4: 12 tiles generated.
- s4_00: first draw accepted (Microsoft, ZIPPER, INTERACTIVE sharp and clean).
- s4_01, s4_02: first draw accepted (title letters sharp with realistic marble fill).
- s4_03: first draw accepted ('The Tall Tales' and 'and' razor sharp).
- s4_10, s4_11: first draw accepted ('SON OF THE EMPIRE' clean, metal panel rivets sharp).
- s4_12: first draw accepted (harbour water reflections clean).
- s4_13: 2 attempts (seed 777 grew shifted letters; seed 12345 with targeted prompt clean and exact).
- s4_20..s4_23: first draw accepted (plain water and black bar clean).
Result: seamless 3200x2400 redraw with all text crisp and authentic.
