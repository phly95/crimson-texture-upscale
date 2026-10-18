# bliteoff
Diagnosis: original is a flat front view of a caged lamp that fills ~90% of the frame width and runs off the bottom edge (alpha opaque inside, cut flat at bottom). Old redraw (retry:2) drew a smaller free-standing capsule with a rounded base, shrunk away from the edges, and the gaps between the cage bars keyed out as transparent holes (coverage 0.80 -> 0.45).
Cause: old prompts never said the lamp must touch top/bottom edges; background-green instructions made FLUX leave green between bars.
Fix: guide = original colours on pure green, hard silhouette (guides/bliteoff_hard.png); prompt (prompts/bliteoff_a1.txt) states fills 90% width, dome touches top, cut flat at bottom, NO green inside the silhouette, bars gaps show dark interior.
Result: attempt a1, all 3 seeds pass (diff 0.014-0.022, iou 0.97-0.98). Chosen seed 2. Pure FLUX.
