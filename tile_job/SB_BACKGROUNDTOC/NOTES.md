# SB_BACKGROUNDTOC notes (tiled FLUX 4x, per-tile prompts)
Final prompts: tile_job/SB_BACKGROUNDTOC/prompts.json. Assembly: seam_assemble.py + seams_s2.json / seams_s4.json.
Lettering in the picture:
- Round gold coin badge: 'ROGUE & Co' (top rim), 'PILOT' (across airplane wings), 'FLIGHTS' (lower left rim), 'LOG BOOKS' (lower right rim)
- Cream paper luggage tag: 'PROPERTY OF:' (typewriter capitals), 'N. ZACHARY' (dark grey handwritten pencil lettering)

## Lessons
- In s4, the luggage tag spans across tile columns s4_20 and s4_21. Initially s4_21 had 'no lettering' in its prompt, which blurred the tail of the 'Y' where the tiles blended. Describing 'PERTY OF:' and 'ZACHARY' explicitly in s4_21 produced matching sharp letters in both tiles.
- Placing the vertical seam at base x=1014 with ramp width 4 puts the transition cleanly in the blank cream paper between the finished 'Y' stroke and the inner tag border, completely avoiding letter ghosting.
- The spine gutter at x=1600 provides a natural vertical seam dividing the leather book cover from the cream paper page.

## Per tile
- s2: 4 tiles generated and assembled cleanly with seam at x=800, y=600.
- s4: 12 tiles generated.
- s4_20: first draw accepted (gold coin badge and name tag text sharp).
- s4_21: 2 attempts (first had blurred text in tag overlap; second with exact text prompt sharp).
- All other s4 tiles: first draw accepted (embossed leather, pencil sketch, blank paper, tape clean).
Result: seamless 3200x2400 redraw with all lettering sharp and authentic.
