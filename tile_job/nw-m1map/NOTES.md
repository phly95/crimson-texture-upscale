# nw-m1map notes (tiled FLUX 4x, per-tile prompts)

Final image: tile_job/nw-m1map_4x_text.png (3200x2400)
UID: rimage_nw-m1map__fb82b2f3
Assembly: seam_assemble.py + seams_s2.json / seams_s4.json

## Lessons
- s2_00, s2_01, s2_10 were kept from previous run (evaluated good).
- s2_11 on first attempt drew a hallucinated dial in the bottom right corner over '1456' because the dial was described ambiguously. Refined prompt to explicitly specify only one dial at top-left, and forbidden any dial at bottom-right. Redrawn with seed 77, perfectly clean.
- s4 tiles (12 tiles):
  - s4_00: FORRER TUNA COMPANY logo and paragraphs sharp on attempt 1.
  - s4_01: KE and ruler ticks sharp on attempt 1.
  - s4_02: Dial (0, 6/9, 18, 27) and grid crossing sharp on attempt 1.
  - s4_03: JE and diagonal line sharp on attempt 1.
  - s4_10: ROCKY PRINTING CO. and compass rose sharp on attempt 1.
  - s4_11: KG and diagonal line sharp on attempt 1.
  - s4_12: Dashed coastline and ruler lines sharp on attempt 1.
  - s4_13: JG, 136, 311 sharp on attempt 1.
  - s4_20: Scale bars (kilometers/miles), CS wings/skull badge, NEW YORK, NY sharp on attempt 1.
  - s4_21: KG, 450 sharp on attempt 1.
  - s4_22: Ruler line and stain sharp on attempt 1.
  - s4_23: JG, 311, 1456 sharp on attempt 1.
- Seam assemble:
  - s2 seams: x=800, y=[460, 600].
  - s4 seams: x=[900, 1620, 2350], y legend=[915, 1400], y map=[960, 1600].
- Remaining defects: tiny paper grain variation across open water, letters in small paragraphs have very minor typographic variations typical of FLUX, but all readable and faithful to original layout.
