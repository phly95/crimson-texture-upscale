# dirtpatch  (HYBRID: old FLUX colours + original alpha)
Diagnosis: dust/dirt ground decal, original alpha soft and translucent (78% partial alpha, 0.54 mean: soft vertical gradient plus irregular blobby outline). The old redraw (klein:1) has the right outline (iou50 0.955) but is fully opaque with a hard edge and a faint dark/yellow rim (cov 0.64 vs 0.54, diff 0.145). Nothing to re-generate: FLUX cannot express the translucency.
Fix: alpha_tools.hybrid(pack redraw, original alpha, smooth 1.0): same texture, the original's soft alpha, colours bled outward (no rim). A pure-shape variant (alpha_tools.modulate, crisp edge x original translucency) also passes (diff 0.04) but kept a slight edge rim; the full hybrid is cleaner on dark/light/green.
Result: diff 0.006.
