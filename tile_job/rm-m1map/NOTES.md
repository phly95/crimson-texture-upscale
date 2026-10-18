# rm-m1map notes (tiled FLUX 4x, per-tile prompts)

Final image: tile_job/rm-m1map_4x_text.png (3200x2400)
UID: rimage_rm-m1map__567d2eaf
Assembly: seam_assemble.py + seams_s2.json / seams_s4.json

## Lessons
- Distinct legend column: ROCKY PUBLISHING COMPANY with small pine trees, DENVER, COLORADO, MODERN PRINTING, Caldwell, Idaho 92412, and AA Weekly badge.
- s2 tiles:
  - s2_00, s2_01 kept from initial run.
  - s2_10 and s2_11 generated and accepted cleanly.
- s4 tiles:
  - s4_00: Early draw grew duplicate MODERN PRINTING lines near the bottom; restricted s4_00 to only the first two lines of the Pirates Society paragraph -> clean.
  - s4_10: Model hallucinated a compass rose in the text area and faint roads in the narrow map margin; refined prompt with negative guidance ("NO compass rose in this tile", "NO roads or tracks in map strip") -> regenerated with seed 77, perfectly clean.
  - s4_20: Compass rose, CS badge, AA Weekly logo, and scale bars rendered sharp.
  - Map tiles (s4_01, s4_02, s4_03, s4_11, s4_12, s4_13, s4_21, s4_22, s4_23): all railway tracks, rocky mountain contours, turquoise lake and ring stains rendered sharply with no hallucinated labels or symbols.
- Seam assemble:
  - s2 seams: x=800, y=[680, 600].
  - s4 seams: x=[835, 1500, 2240], y legend=[895, 1500], y map=[950, 1550].
- Remaining defects: minor typographic variations in tiny paragraphs, overall alignment and text crispness excellent.
