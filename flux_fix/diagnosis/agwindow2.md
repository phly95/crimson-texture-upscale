# agwindow2
Diagnosis: 32x32 tiling window frame: hairline border on all four sides + a thin diagonal brace lower-left to upper-right, two big empty triangles (cov 0.23). Old redraw (retry:2): border ~2x too heavy with bevel, thick bracket with corner plates and big rivets (cov 0.32, iou 0.66); the prompt said 'very thin' but FLUX renders a frame as a chunky object.
Fix: hard guide (thr 0.3, so the faint translucent border and brace both appear) + prompt with explicit percentages (border ~4% of width, brace ~9%), 'no inner bevel, no corner plates, no large rivets'.
Result: a1 all 3 pass (diff 0.047-0.056, cov 0.275-0.284 vs 0.23, iou 0.83-0.86). Chosen seed 3. Pure FLUX. Brace is a little heavier than the original's translucent one but within tolerance.
