# lk_ngun2
Diagnosis: truss decal that fills the frame (top band full width at the very top, bottom chord running to the bottom edge, diagonal outer right edge from the top-right corner to bottom 56%, thick tapered vertical at 55-68%, four open gaps). Old redraw (klein:1) drew the truss smaller with margins (bottom and left not reaching the frame), a wide plate with holes, extra rivets; borderline fail (diff 0.122 vs 0.12; cov 0.35 vs 0.42, iou 0.72).
Fix: hard guide + prompt that lists members with percentage positions, the exact open gaps, 'no other holes, no round holes, no rivets/plates'.
Result: a1 all 3 pass (diff 0.010-0.011, iou 0.915). Chosen seed 3 (fewest invented bolts). Pure FLUX.
