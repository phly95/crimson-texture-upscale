# PC_BACKGROUND notes (tiled FLUX 4x, per-tile prompts)
Final prompts: tile_job/PC_BACKGROUND/prompts.json. Assembly: seam_assemble.py + seams_s2.json / seams_s4.json.
Lettering in the picture:
- Vintage radio (bottom left): script capital 'R'
- Open book (bottom center): newspaper masthead 'Aloha Daily' with airplane icon
- Purple blueprint (bottom right): 'Name: Grumman', 'Weight Capacity: 2100 lbs', 'Overall Speed: High', 'Structure Integrity: Poor'

## Lessons
- The central window is pure black and remains untouched and pure black across all top and middle tiles.
- Placing the vertical seam at base x=2220 with ramp width 16 cleanly isolates the open book in s4_22 from the purple blueprint in s4_23, avoiding any ghosting between the book edge and the blueprint text lines.
- Seams at x=850 and x=1500 sit inside the black window and blank desk gaps.

## Per tile
- s2: 4 tiles generated and assembled cleanly with seam at x=800, y=600.
- s4: 12 tiles generated.
- s4_22: first draw accepted ('Aloha Daily' newspaper masthead sharp).
- s4_23: first draw accepted (all blueprint hand-lettered text lines legible).
- s4_20: first draw accepted (script 'R' on radio, US map sharp).
- All other s4 tiles: first draw accepted (black window, walls, equipment clean).
Result: seamless 3200x2400 redraw with all authentic desk lettering preserved.
