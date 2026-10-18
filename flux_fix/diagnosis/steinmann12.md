# steinmann12
Diagnosis: portal frame filling the frame: lintel from the top edge to 22%, base beam 78%-100% to the bottom edge, thick diagonal struts (~12% wide) to the bottom corners, empty corners/sides and trapezoid opening (cov 0.58). Old redraw (klein:1): thinner struts, margin above the lintel and below the base (neither touched the frame edges), opening too wide (cov 0.50, iou 0.80, diff 0.122, borderline).
Fix: hard guide + prompt with percentage layout, 'no margin: beams touch the top, bottom and side edges', strut width ~12%.
Result: a1 seeds 1,2 pass (diff 0.017-0.019, iou 0.96), seed 3 fails (left a margin: the 'touch edge' instruction is not always obeyed). Chosen seed 1. Pure FLUX.
