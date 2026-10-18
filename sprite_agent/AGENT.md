# Brief: redraw soft effect sprites from scratch, in code

You are redrawing 18 small effect sprites from the 2000 game Crimson Skies (water ripple, splashes, fires, surf and wake strips, a bullet-hole decal, hit flashes) as HD
versions, 8x the original resolution. The engine draws them with alpha blending, so the ALPHA CHANNEL is as important as the colours. A previous AI redraw
failed exactly there (it invented new shapes and changed how much of the picture is transparent). You draw each sprite as a Python recipe using the toolkit in `fx.py`,
render it with explicit alpha, and check it with `sprite_tool.py`, which compares your result with the original at the original's size.

Work only in this directory (/home/deck/workspace/csvm-hd/sprite_agent). Python: `/home/deck/workspace/csvm-hd/venv/bin/python` (numpy, opencv, scikit-image, skia-python, PIL).

## What you have per sprite (`sprites/<name>/`)
- `orig.png`: the ground truth, tiny (32-128 px). It defines the shape, size on the canvas, soft falloff, colours and overall look. Study it zoomed and its alpha channel.
- `ref_upscale.png`: the original resized to the target size (blurry; use it to read positions and colours, e.g. `fx.load(path)`).
- `prev_redraw.png`: the old, rejected AI redraw. Look at it for DETAIL VOCABULARY only (what fine texture might look like); do not copy its shape or alpha.
- `meta.json`: target size. Your `draw.png` must be exactly that size.

## The loop (per sprite)
1. Look: open `orig.png` and `ref_upscale.png` with the Read tool (it shows images). Describe to yourself what the thing is, what its alpha looks like, its palette
   (`fx.palette`, `fx.alpha_stats`, `fx.radial_profile`) and whether it tiles (strips such as srf*, wakefront*, falls are likely tileable: edges must wrap).
2. Write `recipes/<name>.py` that builds the sprite with `fx` and saves `sprites/<name>/draw.png`. Read the docstrings in `fx.py` first. Build it in layers like a painter:
   base shape/silhouette, colour ramp, then multiple scales of noise and warping for fine structure, wisps, highlights, a soft alpha falloff that follows the original's.
3. Run it, then `venv/bin/python sprite_tool.py check <name>`. Read the printed gates AND open `sprites/<name>/sheet.png` (original on top row, yours on the middle row, each on checker,
   black, white; bottom row = alpha of original, alpha of yours, difference). Also open `tile2x2.png` when it is printed.
4. Fix what the gates say, and fix what YOU see: wrong kind of thing, mushy blur at 8x, visible noise grid, banding, hard clipped edges, too-clean computer-graphics look, colours not the original's.
   Iterate. Typical numbers: numeric gates in 2-6 rounds, then more rounds on looks.
5. Only when the gates pass AND the sheet looks right to you (it should read as the same sprite as the original, with convincing extra fine detail, not a different picture),
   run `sprite_tool.py submit <name>`.

The gates are necessary, not sufficient. They are measured on a reduced copy, so a sprite can pass and still look bad up close; judge the full-size `draw.png` too (open it).
Never edit `orig.png`, `sprite_verify.py` or `sprite_tool.py` (they are checksummed), and never copy `ref_upscale.png` pixels as the final colours: the point is real fine detail.

## Order and discipline
- FIRST get exactly ONE sprite right: `ripple`. Make it the best you can, with good looks, not just a passing number. When it is submitted, write `NOTES.md` with what worked
  (parameter ranges, tricks, pitfalls, how you matched alpha) so the rest go faster and so the main agent can learn from it.
- THEN do the rest in the order `sprite_tool.py list` shows, reusing your notes and helper code (put reusable pieces in `recipes/common.py`; `fx.py` itself you may extend with new
  functions but not break existing ones).
- Write `STATUS.md` after every sprite: name, attempts, what you did, anything you doubt. If a sprite still fails after about 12 attempts, write BLOCKED with the reason and move on;
  come back to it at the end.
- Before you finish: run `sprite_tool.py list` (everything SUBMITTED or explained in STATUS.md), re-run `check` on every `out/*.png`, and make `out/_overview.png`
  (a grid of original-upscaled vs your final on a checker, with names) and open it yourself. Fix anything that looks off.
- Final message: counts (submitted / blocked), one line per blocked sprite with the reason, and what you are least sure about.

You may be resumed later with feedback from the project owner about specific sprites. When that happens, read the feedback, fix those sprites, re-check, re-submit
(a re-submit overwrites `out/<name>.png`), update STATUS.md, and answer briefly.

## Batch 2: structures, wires and shadows (added after batch 1)
17 more sprites (`sprite_tool.py list` shows them after the effects; group `structures` in meta.json): `thin_shadow abld_shadow firetruck_shadow mptrailer_shadow agyro_busshadow
bridge_cables bbridgecables lightpole cable_insulator cnine_wire1 rail1 doublewire ladder_01 rope_bridge zep_cable01 grate01 eiffel1`. They are not effects: they are
crisp man-made geometry (cables, girders, trusses, rails, ladders, grates, insulator stacks) and shadow silhouettes/bands. The FLUX redraws of these either invented shapes
(`cable_insulator`, `rail1`, `cnine_wire1` changed layout) or lost the original's soft alpha edge. The originals are tiny (16-128 px) and block-pixelated: draw the SAME object
with the SAME layout, proportions, colours and alpha, as a clean HD version, as if the original artist had painted it at 8x.
- Draw geometry with skia paths/rects/lines (`fx.paint`), anti-aliased, with the real edges in the right places; sample positions and widths from `orig.png` (look at it enlarged,
  and read pixel columns/rows with numpy) instead of guessing. Then add material: metal, rust, paint, rivets, grain, wear, subtle shading (gradients along bevels,
  `fx.fbm` for grime), keeping overall colours within the original's.
- Shadows (`*_shadow`, `thin_shadow`) are soft black or grey shapes: match the silhouette and the falloff of the original's alpha exactly (blur the vector shape with `fx.blur`
  to the right softness); no texture, no detail, no halo. Flat or nearly flat originals (`abld_shadow`, `thin_shadow`) stay flat: a clean gradient/fill at the right values.
- Many of these tile or are placed repeatedly (rails, cables, doublewire, grate01, ladder_01): the tile 2x2 preview must show no seam, and bars/cables must continue across
  the edges at the same positions as the original.
- Gates for this group: the silhouette gate `iou50` is 0.70 (thin lines can't reach 0.85 after reduction); everything else is as before. Still look at every sheet.
- Work in the listed order, submit each when gates pass and it looks right, update STATUS.md and NOTES.md (add a "structures" section), and finish with a refreshed
  `out/_overview.png` including these.

### Note from the project owner on `eiffel1` (reported while you work)
The previous AI redraw of `eiffel1` has green/dark-olive fringing: a dark olive rim runs along every edge of the truss against transparency (green-screen spill). Yours must have none: the
outermost pixels along the silhouette take the colour of the girder next to them (anti-aliased, no outline). `sprite_tool.py check` now measures this as `rim_hue` (limit 10) for every
non-effect sprite and prints the fix when it fails. Look at the edges of your result enlarged on a MAGENTA background, not just checker/black/white, before submitting any cut-out.
