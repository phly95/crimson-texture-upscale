# SB_BG_Y notes (tiled FLUX 4x, per-tile prompts)
Prompts: prompts.json. Assembly: tools/seam_assemble2.py + seams_s2/s4.json (s2: x 1006 ramp 12, y 560 ramp 100 so the two header rules come only from the top tiles; s4: x 745/1577/2288, y 856/1488 ramp 80).
Lettering: 'Pacific Telegram Company' (dark grey serif inside nested ellipse logo with palm-island emblem). Everything else blank blue paper.
## Lessons
- Logo text spans tiles: each tile prompt names the exact fragment it holds ('Pa' / 'Pacific Telegram Com' / 'am Company'). All exact.
- s4_03: my prompt said "right ends of two ellipses" at the top-left, but the ref there is blank paper -> model drew a rounded double box. Check where in the tile things sit; for blank upper parts say "upper part plain paper, nothing on it" and place the rules by position ("in the middle of the crop").
- Rules at the top of a lower-row tile get blurred when blended; put the y seam below them (y 560 at s2) so rules come from the top-row tile.
## Per tile
- s2: 1 attempt each. s4: 1 attempt each except s4_03 (2).
Imperfect: ~1px thickness step in the lower grey rule near x 2300 (seam column).
