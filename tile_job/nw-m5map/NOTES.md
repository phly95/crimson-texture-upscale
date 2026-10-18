# nw-m5map notes (tiled FLUX 4x, per-tile prompts)

Final image: tile_job/nw-m5map_4x_text.png (3200x2400)
UID: rimage_nw-m5map__b7671a1b
Assembly: seam_assemble.py + seams_s2.json / seams_s4.json

## Lessons
- Reused COLOSSAL / Paper and Timber Company legend column prompts from nw-m2map.
- Map area has no text, numbers or dials: all map tiles explicitly instructed with "NO lettering, NO numbers, NO labels, NO symbols anywhere in this tile".
- All s2 tiles (s2_00, s2_01, s2_10, s2_11) and all 12 s4 tiles accepted on attempt 1 with zero hallucinated labels or shapes.
- Seam assemble:
  - s2 seams: x=800, y=[450, 600].
  - s4 seams: x=[835, 1500, 2240], y legend=[895, 1550], y map=[950, 1550].
- Remaining defects: minor typographic variations in tiny paragraphs, coastlines and grid lines completely continuous and sharp.
