# PX_BACKGROUND notes (tiled FLUX 4x, per-tile prompts)
Final prompts: tile_job/PX_BACKGROUND/prompts.json. Assembly: seam_assemble.py + seams_s2.json / seams_s4.json.
Lettering in the picture:
- Top blueprint header: 'Form AC-4523' (thin pale lilac hand-lettered type in table cell)
- Bottom blueprint margin: 'SCALE 1"=10\'0"' (thin pale lilac pencil handwriting)

## Lessons
- In s2, quoting explanation letters like '(S-C-A-L-E, ...)' caused the model to literally draw 'S-C-ALE' as extra text. Quoting only 'SCALE 1"=10\'0"' directly fixed it.
- The top header cell 'Form AC-4523' starts at base x=840. Placing the vertical seam at x=760 with ramp width 20 places the seam in the pure empty grid before the header cell starts, keeping 'Form AC-4523' 100% in s4_01 with zero ghosting.
- The scale line starts at base x=950, so seam at x=760 is also completely before the scale line, keeping it 100% in s4_21.
- Seam at x=1550 cleanly divides the blueprint sheet from the clipped cream paper sheet.

## Per tile
- s2: 4 tiles generated. s2_10 had extra letters from descriptive spelling prompt; prompt cleaned and redrawn sharp.
- s4: 12 tiles generated.
- s4_01: first draw accepted ('Form AC-4523' sharp).
- s4_21: first draw accepted ('SCALE 1"=10\'0"' clean).
- All other s4 tiles: first draw accepted (blueprint grid lines, blank paper, skull watermark seal clean).
Result: seamless 3200x2400 redraw with all technical lettering crisp and clean.
