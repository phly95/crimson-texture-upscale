# loadframe notes (tiled FLUX 4x, per-tile prompts)
Final prompts: tile_job/loadframe/prompts.json. Assembly: seam_assemble.py + seams_s2.json / seams_s4.json.
Lettering in the picture:
- Compass rose at bottom left: 'N', 'E', 'S', 'W'
- Bottom margin: 'Copyright 1937 Universal Air Freight Co. Route planning _SERIES 37_EE'

## Lessons
- Almost all tiles are large blank white panels with cream paper margins and thin dashed double lines. Placing seams across the middle of the flat white panel at y=800, y=1500 and x=800, x=1500, x=2250 yields clean blends.
- The compass rose and copyright line are fully contained in s4_20 with no seam crossing any text.
- Describing the panel as 'large flat white rectangular panel' and the margin as 'soft cream paper margin' preserved the original look without hallucinated marks.

## Per tile
- s2: 4 tiles generated and assembled cleanly with seam at x=800, y=600.
- s4: 12 tiles generated.
- s4_20: first draw accepted (compass rose, letters N/E/S/W, and full copyright line sharp).
- All other s4 tiles: first draw accepted (flat white panel, dashed borders, and deckled paper margins clean).
Result: seamless 3204x2404 redraw with all text and borders crisp.
