# IA_BACKGROUND notes (tiled FLUX 4x, per-tile prompts)
Final prompts: tile_job/IA_BACKGROUND/prompts.json. Assembly: seam_assemble.py + seams_s2.json / seams_s4.json.
Lettering in the picture:
- Top-left badge: 'SPICY!' (chunky orange letters), 'Air', 'Tales' (white cursive script), 'AIR ACTION WEEKLY' (top rim curved capitals), 'PRESENTS' (bottom rim curved capitals)
- Top-right banner: 'HEY' (curved bold black capitals), 'PILOT!' (bold black capitals with underline), winged skull emblem in black box
- Bottom emblems (left and right pages): winged disc with 'SPICY!', 'Air', 'Tales'

## Lessons
- Placing the vertical seam at base x=1600 directly down the dark spine gutter of the book allows a clean split between left and right pages with zero text crossing the seam.
- The top-left badge and top-right 'HEY PILOT!' are contained fully within their respective column tiles (s4_00 and s4_02) so no seam ramps cross the lettering.
- The horizontal seams placed at y=800 and y=1500 through the large blank rectangular panels avoid all text and icons entirely.

## Per tile
- s2: 4 tiles generated; clean first draw for s2_00, s2_01, s2_10; s2_11 retry on server contention then clean.
- s4: 12 tiles generated.
- s4_00: first draw accepted ('SPICY! Air Tales', 'AIR ACTION WEEKLY', 'PRESENTS' sharp).
- s4_02: first draw accepted ('HEY', 'PILOT!', pilot face shading and wood table sharp).
- s4_01, s4_03, s4_10..s4_13, s4_20..s4_23: first draw accepted (paper texture, wood grain, dashed boxes clean).
Result: seamless 3200x2400 upscale with all authentic menu lettering preserved.
