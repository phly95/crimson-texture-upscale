# Water flipbook HD rebuild

Run: `venv/bin/python -I make_water.py GROUP` (defaults: amp 1.5, sigma 4, mg 0.5, tf 1.25, rolls 24; C4 was run with `--amp 1.1`),
then `water_check.py GROUP out/GROUP`, `tile_check.py GROUP`, `crop_cmp.py GROUP 0 out/GROUP 1.0`, `film.py GROUP`.
Outputs: out/GROUP/wtr00000..15.png (512x512 RGB), plus strip.png / diff_strip.png / tile_check*.png / crop_cmp.png / film.png.

## What the originals are
- Very low contrast (per-channel std 2-4 of 255, 2-5 distinct values per channel), diagonal "/" streaks.
- Frames k and k+8 are bit-identical: it is an 8-frame loop shown twice. The output makes 8 frames and copies them to k+8.
- Motion is real but not a rigid scroll: consecutive frames match with a drift of about (+1.4, +0.3) px/frame (Farneback on lightly blurred frames
  explains ~70% of the MSE vs ~0 for no motion), lag-4 frames are anti-correlated. Sum of the 8 flows = ~(96, 22) HD px, yet frame 8 == frame 0
  (the texture is quasi-periodic along the drift), so detail carried by the flow cannot simply loop.

## Method
1. Detail layer: from the approved HD frame 0 (hd_current/wtr00000): fine band (frame minus Gaussian sigma 4 px) at gain 1 plus mid band
   (sigma 4..12) at gain 0.5, from the non-periodic frame using reflect-mode blurs. Made tileable with `periodic_blend`: a variance-preserving
   cos/sin cross-fade of the band with itself rolled by half a period along each axis (centre = original, wrap edge = the rolled copy, so no
   structure is cut). Moisan periodic+smooth decomposition was tried first: it makes the pixel seam zero but leaves a visible content cut
   (a "+" / V shaped discontinuity of ridges) that the check cannot see - dropped (`--period moisan` still available).
   Local detail energy is flattened (`--flat 0.8`, sigma 40) so no part of the tile (e.g. the busy top of the AI frame) is special; then the layer
   is rolled by the offset (of 24 random ones) that gives the lowest worst-frame wrap-seam score.
2. Motion: Farneback flow between consecutive ORIGINAL frames (wrap-padded, sigma 1.5 pre-blur, flow blurred 2 px), composed into T_k (frame-0 content ->
   frame k). Detail frame k = cos(pi k/16) * D(x - T_k) + sin(pi k/16) * D(x - T_k + T_8): the second copy is the same warp shifted by the loop drift
   T_8, so at k = 8 it is exactly D again (loop closes) and frame 0 is exactly the source detail. The cross-fade is variance preserving.
3. Frame = Gaussian(3px)(wrap-bicubic upsample of original frame k)  +  gain_k * detail_k, then 3 passes of partial back-projection
   x += G_csig(up(orig_k - box8(x))). This pins the mean colour and large shapes to the original (de0 <= 0.1) and cancels the part of the
   detail that leaks into the 8x8 box average. csig (smoothing of the correction) is the knob for the check's tflicker: auto-bisected per group
   (smallest 0.8, largest 10) to the largest value with tflicker <= 1.25 (more csig = more mid-band body, more flicker at the original's scale).
   gain_k is iterated so the Laplacian variance is equal across frames (tdetail ~ 0).
4. Everything uses wrap boundaries (wrap pad+resize, FFT-free `gaussian_filter(mode="wrap")`, `grid-wrap` spline warps), 8-bit rounding at the end.

## Results (water_check, all pass)
| group | de0 | de_mean | recon | seam (limit) | tflicker | tdetail |
|---|---|---|---|---|---|---|
| C5 | 0.05 | 0.06 | 0.25 | 0.97 (1.30) | 1.25 | 0.00 |
| C4 | 0.01 | 0.02 | 0.15 | 0.87 (1.30) | 1.28 | 0.00 |
| C3 | 0.01 | 0.01 | 0.25 | 0.87 (1.38) | 1.25 | 0.00 |
| A_C1B_C2_C2B | 0.09 | 0.11 | 0.26 | 0.90 (1.40) | 1.25 | 0.00 |

## What failed / caveats
- First version (full back-projection, high-pass sigma 6, unblurred base): passed but looked like upscaled dither blotches plus ridge lines; the base is now
  blurred 3 px and the mid band was added.
- Seam score > limit in the first runs for C3/C4/A (1.3-1.5): not a real seam but uneven detail density near the tile edge rows. Fixed by energy
  flattening + roll search. The score is a noisy statistic (+-0.15 between frames); margins are 0.3-0.5 now.
- C4's HD frame 0 is much more contrasty than its flat original; with amp 1.5 csig hit the 0.8 floor and tflicker was 1.39 (still <1.5), so C4 uses amp 1.1
  (tflicker 1.28, csig at floor). C4 detail is therefore the weakest relative to the approved look.
- Detail amplitude is ~70-100% of the approved frame 0's fine band but well below the approved frames 3-5 (which have much stronger glints and waves). The
  large-wave "body" of the approved frames is deliberately attenuated: it would otherwise flicker (tflicker) against the original's 8-px-scale motion.
- Detail is a single source frame (frame 0), so there are only one tile's worth of ripples; specular glints of the other approved frames are not used.
- The detail is rolled relative to the original's frame-0 structure (roll search), so any alignment between the AI frame-0 and the original frame 0 is lost.
  Crossfade frames (k=3..5) show a mild double-image of the ripple lines.
- Colour speckle (AI chroma noise) survives in the detail, visible only brightened.
