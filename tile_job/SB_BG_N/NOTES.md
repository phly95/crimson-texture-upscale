# SB_BG_N notes (tiled FLUX 4x, per-tile prompts)
Prompts: prompts.json. Assembly: tools/seam_assemble2.py + seams_s2.json / seams_s4.json (x seams at min-diff columns, ramp 32, y seams 920/1544).
Lettering: blackletter masthead 'Business Journal Weekly' (cut by the top edge), small telephone icon at left. Rest is blank crinkled paper.
## Lessons
- First s2 draw gave the black letters a light halo/drop shadow; "flat black ink ... printed flat on the paper, crisp edges, no shadow and no glow" removed it.
- Blackletter has no ink-free column in the seam range; seam at the column where the two tiles differ least (tools: min mean abs diff), ramp 32: fine.
- "one soft horizontal fold band" in the prompt made s4_11/12 invent a thin crease line across the middle of the sheet; "smooth soft shading, wide soft pale band across the upper part of the crop, no creases or lines" fixed it.
## Per tile
- s2_00/s2_01: 2 attempts (halo). s2_10/11: 1. s4: 1 attempt each except s4_10..13 (2, invented crease).
Imperfect: grey torn strip at the left edge is smoothed to a straight narrow band; telephone icon has a slightly different top.
