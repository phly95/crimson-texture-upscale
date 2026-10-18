# Lettered capsule buttons: style definition (pass 2)

One renderer, `button.py`, draws every button. All numbers live as named constants at the top of `button.py`; this file explains
them. Lengths are in ORIGINAL pixels (x5 at the 5x target) unless stated otherwise.

## Inputs measured from each original (`button.measure(stem)`)
- **Lettering mask**: state 3 (orange) minus state 4 (red), green channel > 60. The plate is identical in those two states, so only
  the letters differ. States are aligned first (EM_ sheets step their plates by about 2 rows per state). Flat wide streaks (2 rows
  or less) are dropped. This gives the text box, cap height, word boxes (column runs split at the n-1 widest gaps) and letter centres.
- **Support bar end**: the end of the longest light-grey run in the bottom 30% of state 2.
- **Rim rows**: per state, the top rim highlight is detected as a narrow bright peak (e.g. `43 96 43`). The bottom rim row is fitted
  by matching the mapped master's vertical luminance/alpha profile to the original (bar columns and no-bar columns, text rows
  ignored, rows below the plate's own bottom ignored). The rim distance is shared by all 4 states.
- **Word breaks**: `LABELS` in button.py (the sprite names drop the spaces).

## Plate (`fit_plate`): master_plate.png, never redrawn
- **Frame weight** `frame_scale()`: the originals use three frame weights, so the master is used at `KM*kb` with kb = **0.7** for
  32-38 px states, **0.8** for the MM_ 42 px main-menu family, and **1.0** for the other 42-50 px buttons.
- **Width**: end caps (master x < 178, x >= 972) and the bar-end zone (700-822) are untouched. The strip above the bar (178-700) and
  the strip right of the bar end (822-972) are cropped or extended by quilting their own columns with a minimum-error vertical
  seam (40 px overlap, 1.2 px feather). The bar highlight ends exactly where the original's does.
- **Height**: a five-band piecewise map. The top rim (master rows 0-40), the side brackets (74-124) and the bottom rim + bar
  (140-208) stay at frame scale. The two parchment bands take up the difference (vertical low-pass when squashed). Rows go
  through `cv2.remap` in premultiplied RGBA, so edges stay anti-aliased with no fringe.
- **Mottling**: the original's own low-frequency parchment pattern (Gaussian sigma `MOTTLE_SIGMA` = 1.0, strength
  `MOTTLE_STRENGTH` = 0.8, Lab) is carried onto the panel, so each plate keeps its original's blotch layout. The original's
  letters, plus 2 px and their shadow, are excluded. The master's grain, filigree and frame stay as they are.
  For a new button without an original, this step is skipped automatically.
- **State 1 (grey parchment)**: only the panel changes (soft chroma-based panel mask). One fixed Lab mean/std transfer, measured
  over all 56 originals: brass L,a,b mean (60.4, 3.86, 34.2) / std (7.51, 0.99, 3.39) maps to (55.0, 0.71, 15.6) / (10.36, 0.73, 2.15).
  States 2-4 use the brass as is.
- **Sheet edges**: the checker treats these sheets as tiling. The two outer columns are averaged, and the first and last rows are
  pulled 20-50% towards each other (the smallest amount that passes). This is 1 px at 5x.

## Lettering (identical on every button)
- **Font**: Pathway Gothic One (`fonts/PathwayGothicOne-Regular.ttf`, OFL), with strokes thickened by `WEIGHT` = 0.17 px per side.
  Chosen by eye against the originals' condensed, rounded-terminal face. Other condensed faces tested (Yanone, Antonio, Saira EC,
  Oswald, Sofia Sans EC, Bebas) changed the colour metric by only ±0.3.
- **Size and position**: the cap height equals the original's lettering rows. Each word is centred on the original's word box.
  Glyphs are pulled halfway (`ALIGN` = 0.5) towards the original letter centres where letters are separable.
- **Spacing**: tracking is fixed at `TRACK` = 0.09 x cap height (allowed range 0.04-0.16). One horizontal glyph scale per button
  (`SX_RANGE` 0.88-1.15) absorbs the remaining width difference. Long labels come out slightly condensed, short ones slightly wide;
  the spacing rhythm is the same everywhere.
- **Layers per state**, bottom to top:
  1. dark halo: blur 2.0, gain 1.8, opacity 0.80, offset (0.4, 0.6)
  2. coloured glow: normal blend, blur 0.9, gain 1.8
  3. soft drop shadow: offset (0.9, 1.6), blur 1.1, gain 2.2 (a dense core with a soft falloff)
  4. tight dark contour: offset (0.15, 0.25), blur 0.4, opacity 0.85
  5. letter fill: vertical gradient top / middle / bottom over the cap height, plus a bevel (strength 0.35, light from top-left)

| state | fill top / mid / bottom (sRGB) | glow (colour, opacity) | shadow colour, opacity |
|---|---|---|---|
| 1 normal | (212,212,210) / (180,180,179) / (160,160,158) silver | none | (14,11,8) .95 |
| 2 hover | (255,238,176) / (255,217,128) / (252,204,106) pale cream-gold | (255,214,120) .30 | (30,16,2) .95 |
| 3 active | (255,176,74) / (255,150,34) / (246,74,0) orange to red | (236,104,10) .55 amber | (26,4,0) 1.0 |
| 4 | (255,74,66) / (248,38,34) / (232,8,8) red | (214,28,16) .55 red | (22,0,0) 1.0 |

The fill colours are measured from the originals' letter cores. Median core colours across all 56 originals: (178,178,178),
(254,217,128), (255,150,35) and (246,30,30). The cream of state 2 is the original's own colour; it is not pushed towards yellow-orange.

## Reusing it for a new button
Without an original, build a spec by hand and call `render_spec(spec)`:
`dict(label, W, H, sh, plate=dict(x0,x1,y0,y1), bar_end, rims=[(tc, bc, kb)]*4, text=dict(x0,x1,y0,y1, words=[(a,b),...]))`.
Copy `rims` from a button of the same size class:
- 32 px: (2.0, 28.7, 0.7)
- 42 px: (2.0, 37.2, 1.0)
- 50 px: (2.0, 45.2, 1.0)

## Checks
- `consistency.py` measures, per state, the median letter colour, plate mean/std and shadow depth/softness on every rendered button,
  lists outliers, and writes `out/_overview.png` and `out/_consistency.png`.
