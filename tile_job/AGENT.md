# Tile-prompt agent brief

Goal: a 4x (3200x2400) redraw of a fullscreen game picture by FLUX.2 klein, tile by tile, where **every tile's prompt says exactly what lettering that tile holds (or that it holds none)**,
so the model neither garbles real text nor invents labels/symbols/creases. You are the writer and the judge. Tool: `venv/bin/python -I tools/tile_job.py` (read its docstring first).
Work dir: /home/deck/workspace/csvm-hd. Look at pictures with the Read tool (it shows PNGs). The klein server is already up (shared; never stop it).

## Loop (per tile; settle all s2 tiles, then `next`, then all s4 tiles)
1. Look at the reference: `tile_job/<NAME>/<stage>/ref_YX.png` (what the model edits) and, for reading text, the quadrant views from `view`. For s2 tiles also look at `tile_job/<NAME>/src.png`
   (the true 800x600 original; crop+enlarge regions with PIL to read small print). Transcribe all lettering in the tile verbatim, note size/colour/font style/position.
2. Write the tile prompt into `tile_job/<NAME>/prompts.json` under the tile id. Good prompts: say what the tile shows in plain words; give each text line verbatim in quotes with where it sits and its look;
   say everything else has no lettering; forbid new labels, numbers, symbols, creases, scratches, wear that are not in the reference; name materials (paper, brass, wood...) and say to keep them as soft/clean as the reference when they are.
   Cut-off words at a tile edge: say so ("the word is cut by the right edge, only 'PAP' is visible").
3. `gen NAME TILE` then `view NAME TILE`; Read view_TILE.png and the quadrant files. Judge: (a) every real line of text is present, spelled right, same place; (b) no invented text, symbols, shapes;
   (c) layout/lines/shading match the reference; (d) sharp, not smeared. If wrong, say precisely what is wrong, refine the prompt (add the missing/forbidden thing, change wording) and/or `--seed N`, and redraw.
   At most 6 attempts per tile; keep the best. A tile with only tiny unreadable print may accept slightly soft letters as long as nothing is invented.
4. After the s2 tiles: `assemble NAME s2`, Read s2/stage.png, then `next NAME`; do the s4 tiles the same way (their references are the s2 result, so text is already 2x; transcribe from it).
5. `final NAME tile_job/<NAME>_4x_text.png`, Read a few 2x crops of the whole result, fix tiles that show seams or problems (regen tile, re-run `final`).
6. Submit: `venv/bin/python flux_fix/submit_fix.py <UID> tile_job/<NAME>_4x_text.png --label "Tiled FLUX 4x, per-tile text prompts"` (UID is given in your task).
Keep `tile_job/<NAME>/NOTES.md`: per tile the final prompt, attempts, what went wrong, what fixed it (the next jobs reuse these lessons). Do not touch anything else (pipeline/, other jobs, services).
Final reply: per image, which tiles needed what, anything still imperfect (name the tile and the defect), time spent.
