# Sprite Redraw Agent Notes

## Overview of the Strategy

Soft-alpha effect sprites in the engine are rendered with additive or alpha blending over dynamic backgrounds. Both the visual texture and the alpha falloff profile must match the original when reduced, while providing rich, believable high-frequency organic structure at 8x magnification.

### What Worked: The Layered Pipeline

1. **Macro Envelope & Silhouette Alignment**:
   - Resampling the original with `cv2.INTER_LANCZOS4` provides a smooth, artifact-free low-frequency spatial envelope (`env_a`).
   - Coordinate transforms: `fx.radius` and `fx.angle` centered at image midpoint, distorted by domain-warped FBM noise (`fx.warp`) along natural flow directions (e.g. tangential and radial vectors for circular ripples/splashes).

2. **Multi-Scale Organic Detail Synthesis**:
   - **Wave Ridges / Capillaries**: `(sin(r_warped * freq + phase) * 0.5 + 0.5) ** p` produces crisp wavefront crests with soft troughs. Frequencies in range 120–300 at 512px.
   - **Foam Filaments / Webbing**: Ridge noise `(1.0 - np.abs(fx.fbm(...) - 0.5) * 2.0) ** gamma` with gamma ~ 2.0 to 2.5 creates cellular foam boundaries, froth filaments, and wispy threads.
   - **Bubbles / Pores**: `fx.smoothstep(0.45, 0.85, fx.fbm(scale=6..10))` creates discrete foam bubble pores.
   - **Specular Droplets / Spray Beads**: Thresholded high-frequency noise `(fbm > 0.80) * (fbm - 0.80) * mult` for crisp micro-beads on crests.

3. **Detail Modulation with Energy Conservation**:
   - Combine detail layers into a zero-centered or 1.0-centered multiplier:
     `detail = base_offset + w1*ridges + w2*bubbles + w3*capillaries ...`
   - Apply modulation weighted by presence: `a_hd = env_a * ((1 - presence) + presence * detail)`.
   - Outskirts fade cleanly without noisy floating specks.

4. **Low-Frequency Residual Calibration (The Alpha Guarantee)**:
   - When downsampled by `Image.BOX` to original resolution, candidate alpha $A_{cand}$ must equal $A_{orig}$ within 0.04 MAE.
   - Residual error $\Delta = A_{orig} - \text{box\_down}(A_{cand})$ is computed at low resolution.
   - $\Delta$ is smoothly upscaled (cubic or linear) and added back to $A_{cand}$.
   - Because $\Delta$ contains only low spatial frequencies (wavelengths $\ge 8$ pixels), adding it adjusts the slow background envelope without smoothing or altering any of the high-frequency 1-2 pixel bubbles, ripples, or highlights!
   - Strict zero-masking is applied where the original sprite has exactly zero alpha, ensuring clean transparent margins.

5. **Shading & Palette**:
   - Compute crest/highlight intensity from the high-frequency detail components.
   - Interpolate between base color, specular highlight, and deeper trough color based on crest intensity.
   - Ensure RGB is zeroed out wherever alpha is 0 (`layer[..., :3] *= (layer[..., 3:] > 0)`).

## Batch 2: Structures, Wires and Shadows

Batch 2 consists of architectural structures, industrial hardware, cables, and vehicle/building shadows.

### 1. Vector Geometry & Crisp Anti-Aliased Silhouettes
- Unlike organic effects (smoke, fire, water splashes), structures have crisp straight or curved edges (`rail1`, `ladder_01`, `rope_bridge`, `eiffel1`, `cable_insulator`).
- Skia vector paths (`fx.paint`) provide mathematically clean lines, rects, and round caps.
- Blending vector geometry with continuous Lanczos envelopes (`cv2.INTER_LANCZOS4`) provides both razor-sharp edges and nuanced specular bevel highlights.

### 2. Vehicle and Flat Shadows
- Shadows (`firetruck_shadow`, `mptrailer_shadow`, `agyro_busshadow`) must be clean, smooth vector-like dark fills with realistic diffuse Gaussian edge falloff.
- Any artificial texture, noise, or grain degrades the shadow look in-engine. Smooth Gaussian filtering (`sigma ~ 1.0..1.5`) removes pixelation steps from tiny low-res originals while preserving clean aerodynamic curves.
- `thin_shadow` and `zep_cable01` have 1D cross-sectional profiles: interpolating the 1D profile across the perpendicular axis guarantees mathematically zero longitudinal seam (`seam = 0.0`).

### 3. Seamless Tiling on Structural Strips & Panels
- **1D Tiling**: `srf*`, `ladder_01`, `cable_insulator` tile vertically (`vtile=True`); `wakefront*`, `doublewire`, `zep_cable01` tile horizontally (`htile=True`).
- **2D Tiling**: `falls`, `grate01`, `eiffel1` tile on both axes (`tile=True`).
- **Seam Metric Pitfall at 8x**: The seam metric `edge / max(inner, 1e-4)` compares the boundary step between row 0 and row -1 against the inner step `np.diff()`. When upscaling 8x, `np.diff()` in smooth interior regions naturally becomes 8x smaller. If the boundary step does not vanish, the seam ratio can artificially explode. Equalizing opposing boundary pixels (`(p[0] + p[-1]) / 2`) across 4–8 margin rows eliminates boundary discontinuity and drops the seam ratio well below the pass threshold.

### 4. Edge Rim Hue Check
- For structure sprites, `sprite_verify.py` checks that outermost solid pixels do not exhibit keying spill or colored halo fringes (`rim_hue <= 10`).
- Material color must be maintained all the way to the edge of the alpha mask, ensuring zero green/olive outline artifacts when composited over skies or building facades.

## Reusable Helpers in `recipes/common.py`

- `calibrate_alpha(a_hd, a_orig, strict_zero=True, iterations=2, tile=False, vtile=False, htile=False)`: runs the box-downsample residual correction supporting seamless periodic boundary wrapping.
- `calibrate_color(rgb_hd, rgb_orig, a_orig, iterations=1, tile=False)`: color calibration ensuring minimal deltaE.
- `periodic_resize`, `vtile_resize`, `htile_resize`: periodic border wrapping upsamplers for tiling textures.
- `polar_flow(W, H, cx=0.5, cy=0.5)`: returns $r, \theta, \vec{t}, \vec{r}$.
- `ridge_noise(W, H, scale, octaves, seed, gamma=2.0)`: generates crisp filaments/ridges.
