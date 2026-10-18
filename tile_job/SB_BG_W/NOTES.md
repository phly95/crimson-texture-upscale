# SB_BG_W notes (tiled FLUX 4x, per-tile prompts)
Prompts: prompts.json. Assembly: tools/seam_assemble2.py + seams_s2/s4.json (x seams min-diff columns 757/1707/2204 ramp 16; y 990 ramp 24 and 1544 ramp 80). Post step: fix_rule.py (see below).
Lettering: struck-through number + '980938-934', 'APR.OFf', blue-ink signature + '9/E/37', 'PROCESSED' frame, red 'SEP 0 8 1 9 3 7', round seal 'BONUM TUM VIDE BATUR CONSILIUM' with clover '13'.
## Lessons
- Seal ring + 'PROCESSED' + date reproduced exactly in all tiles when each tile names which letters are cut by which edge.
- Handwriting: 'exactly the existing strokes ... do not add any further loop in the empty middle of the blue cell' (first s4_00 invented an extra loop). Number line '980938-934' came out smeared on 2 of 3 seeds; tile_best (5 seeds, OCR) picked a clean one.
- Blank-paper tiles: s2_01 and s2_10 first invented a large pale rectangle / torn edge; "whole lower half is plain soft pale paper edge to edge, no boxes, no rectangles, no lines, no torn edge" fixed it. Do not mention a torn edge for tiles whose ref has none (s4_20/21 grew one).
- The thin grey top rule (aliased, 2px) was redrawn by FLUX as a comb of dashes in x<1760 at s4. fix_rule.py blurs the rule along x only (post step) -> smooth grey line.
## Imperfect
- Struck-through number's digits are blurred (like the original, only 'strike' visible). Rule contrast slightly lower than original at x 600-1000.
