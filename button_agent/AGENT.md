# Brief: one consistent look for the game's lettered menu buttons

The game has 59 lettered capsule buttons (`sprites.json` order; names like `PX_B_SELLPLANES`, `MM_B_CAMPAIGN`, `PF_B_AUDIO`). Each original is a sheet of 4 stacked states of one capsule button
(top to bottom: silver-grey lettering on a grey parchment plate = normal, cream/gold lettering on a brass-tan plate, orange lettering, red lettering), 28-50 px per state, with the label baked in.
An earlier AI pass redrew every sheet separately, so every button came out in a different style: different lettering weight and colour (some yellow, some orange, some grungy with burn marks),
different plate texture. The engine author noticed (the original lettering is pale cream with a light glow and a soft dark shadow; the redraws turned it saturated yellow-orange).
Your job: ONE renderer that draws every button, so that all 59 are identical in plate material, frame, lettering font, lettering effects and per-state colours, differing only in size and label text.

## PASS 2 (supersedes the PLATE part below; the owner rejected pass 1's look)
Pass 1 drew its own plates (blocky, pixelated texture) and heavy, unevenly spaced lettering; the owner did not like it (`PF_B_CONTROLS` was the example). New plan:
- **The plate is now a FLUX-drawn, owner-approved blank master: `master_plate.png`** (1104x208 RGBA, the brass state of a 221x42 button, transparent corners, clean steel frame,
  rivet brackets, support bar below, mottled parchment, NO lettering). Its look is final: do not redraw it, do not use any `prev_redraw.png` plate, do not draw a procedural plate.
  `master_plate_orig_state.png` is the original game state it was made from (for reference).
- **You only (1) fit the plate to each button's size, (2) recolour it per state, (3) put the lettering on top.**
  1. Fit: each target (see `sprites/<name>/meta.json`: target = 5x original, 4 stacked states of `state_h` x 5 px height). Per state: scale the master to the state height keeping its aspect,
     then build the width by keeping the left and right end caps (rounded ends with the brackets) untouched and stretching or cropping the plain middle (blend the joins invisibly; the support bar
     below the panel is straight and may be lengthened/shortened like the original's). Corners stay transparent; edges anti-aliased; no rim/outline.
  2. Recolour: states 2, 3 and 4 use the master's brass as is. State 1 is the grey-beige parchment state of the original: recolour only the parchment panel (mask it by position/colour) to the original's
     state-1 colour statistics (measure them from `orig.png`), keep the steel frame as is. The original sheets show exactly which parts change between states.
  3. Lettering (the same font and style on EVERY button, centred in the original's text box, exactly the original's text position/size, even letter spacing, crisp edges, no blocky or doubled
     edges, no outline artefacts): state 1 silver-grey embossed, state 2 pale cream with a light glow and a soft dark shadow, state 3 orange, state 4 red, glow/shadow as in the originals.
     Reuse your font choice from pass 1 only if it matches the originals; check against the originals' letter shapes again.
- **Pass 2 starts with ONE button: `PF_B_CONTROLS`.** Build the renderer, make it right (look at it enlarged on dark and light backgrounds and next to `orig.png` upscaled), `check` and `submit` it,
  write what you did to STATUS.md, then STOP and report (do not start the other 58): the owner reviews it first, and you will be resumed to do the rest.
- State was reset: `sprite_tool.py list` shows everything as todo; pass 1's outputs and notes are archived in `pass1/` (read them only for what worked, e.g. font choice).

Work only in this directory (/home/deck/workspace/csvm-hd/button_agent). Python: `/home/deck/workspace/csvm-hd/venv/bin/python` (numpy, opencv, skimage, skia-python, PIL; `fx.py` is a small layer toolkit,
read its docstrings). You have internet and Bash: you may download a free font.

## Per sprite (`sprites/<name>/`)
`orig.png` (ground truth: layout, text position and size, plate shape, alpha, colours), `ref_upscale.png` (the original at the target size, blurry), `prev_redraw.png` (the old inconsistent AI redraw: look at it
for what the plate material COULD look like at HD, never copy it per button), `meta.json` (target size = 5x original, `state_h`, `label`).

## What to build
1. `button.py`: `render(label, W, H, ...) -> RGBA layer` at 5x, drawing the 4 states stacked. Components:
   - PLATE: capsule frame (dark tubular steel rim, small side brackets/rivets, thin support bar below, as in the originals) around a parchment/brass panel with subtle mottling. Parametrised by size,
     so wide and narrow, 28 px and 50 px tall plates all come from the same code. The same material and the same texture scale on every button.
   - LETTERING: one font for all, matched to the originals' lettering (tall, condensed, slightly squared, uppercase, spaced). Pick it by testing: render each label in candidate fonts (installed:
     fc-list; or download e.g. condensed grotesques) at the original's text height and compare with the original text mask (IoU / position / width). Keep the font that matches best overall, fit each
     label to the SAME text box the original uses (centre, cap height; read them from the originals). Per-state lettering style identical on every button: state 1 silver-grey with a soft emboss, state 2
     pale cream with a light glow and soft dark shadow (the colour the author says the original has), state 3 orange, state 4 red, each with the original's glow/shadow behaviour, measured from the originals.
   - ALPHA: outside the capsule transparent exactly as in the original; clean anti-aliased edge, NO dark/green rim or outline artefacts (the checker measures `rim_hue`).
2. Start with `PX_B_SELLPLANES` (the unit that triggered this), get it exactly right, look at it enlarged on dark and light backgrounds, then `PX_B_CANCELEXPORT` (same size) to prove the renderer
   transfers, then the other sizes. Then run all 59, writing `recipes/<name>.py` (can be three lines calling `button.render`) and submit each with `sprite_tool.py check/submit` exactly like the other agent:
   `venv/bin/python sprite_tool.py check <name>` (gates on the reduced result vs the original: alpha, silhouette iou >= .9, colour dE < 10, rim, tile) then look at `sprites/<name>/sheet.png`, then `submit`.
3. Consistency check you write yourself (`consistency.py`, run it before finishing and fix anything it finds): for every submitted button and state, the median lettering colour, the plate's mean/std
   colour and texture sharpness must agree across buttons within a small tolerance; produce `out/_overview.png` with all 59 in a grid and `out/_consistency.png` showing the same label letters
   taken from different buttons side by side. Open both and look.

## Rules
- The gates are necessary, not sufficient: the buttons must look like the original game's buttons (same attitude, richer HD detail), consistent with each other, and the lettering must be legible and
  crisp at full size. Never edit `orig.png`, `sprite_verify.py` or `sprite_tool.py` (checksummed).
- Write `STATUS.md` after each size class (what you did, doubts), `NOTES.md` with the style definition (font, colours per state, effect parameters) so it can be reused for buttons not in this list.
- If a button cannot pass after ~12 attempts, mark it BLOCKED with the reason in STATUS.md and continue.
- Final message: counts, blocked list with reasons, what you are least sure about. You may be resumed with feedback from the project owner: then fix what is named, re-check, re-submit, update STATUS.md.
