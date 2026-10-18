# CR_BACKGROUND notes (credits page, two columns of small golden text on leather)
Prompts: prompts.json (s2 by mkp.py with the full text of each column, s4 by mkp4.py: per tile the VISIBLE part of each line, cut words as fragments, estimated from word boxes of the 800x600 source). Transcription of all names read from 3x crops of the source (all legible).
Seams: seams_s4.json (seam_assemble2.py): x=916 (gap between column text and tile 01), 1600 wide ramp in the empty medallion zone, 2220; y seams in line gaps (left col and right col have different line pitch -> 3 x zones).
## Lessons
- Giving the whole column text in quotes per tile works: names come out right at 4x (Redraw is nearly verbatim even for 100+ word prompts).
- Tiles whose ref only holds fragments of cut lines (s4_01: 'oy','er','emens,'): the model invents partial words ('Moxoy', 'ion:'). Fix used: paste tile 00's pixels over tile 01's left 300 px (same content, identical text) with a feather.
- A prompt that quotes lines the tile does not contain, or lines near the bottom edge, can produce doubled shifted lines under the last text line (s4_23 first draw); adding 'last line ... below it only plain dark leather' + 4 seeds fixed it.
- Mentioning the skull medallion in text-only tiles makes the model draw it again; mention it only in tiles that show it.
