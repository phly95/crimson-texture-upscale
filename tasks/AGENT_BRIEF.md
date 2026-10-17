# Texture agent brief: CSVM HD texture pack

You are one of many agents each upgrading a small group of textures from the 2000 game *Crimson Skies*
(1930s alt-history, diesel-punk airships and biplanes) so they look good when the game is played at 4K.
Your task id is given in your prompt (`tXXXX`). Work only inside `/home/deck/workspace/csvm-hd`.

## Looking at textures: pictures first, text as the fallback
Try the Read tool on `out/<id>/compare_<uid>.png` (original | result shrunk | result 1:1 crop). If it
shows you the picture, judge by eye: material identity, fine structure, palette, seams. If it instead
returns `[image omitted: this server has no image encoder]`, stop trying to view images and use the text
instruments below for the whole task (they work either way, and their numbers are what the gate checks).
Even when you can see, run `compare_view`/`stats_view` once per unit: a picture will not tell you the
grain angle is 90 degrees or that the band energy is 3x too high. The three text instruments:

```python
print(text_view(uid, cols=48))             # ASCII portrait: 'lum' layout, 'edge' lines+pattern scale, 'sat' colour
print(stats_view(orig(uid), "orig"))       # one line: mean colour, std, grain angle/coherence, band energies, seam
print(compare_view(uid, out_path))         # ORIGINAL | RESULT shrunk | RESULT 1:1 crop, for lum and edge, + a diff map
```
`tools/gate.py` writes `out/<id>/compare_<uid>.txt` — **Read that file for every unit** (it is text).
Reading an ASCII panel is a real skill: a vertical streak is a column of `#`/`-` in a field of `,`;
bricks are a grid of `-` rows; a smooth glow is a graded `#++++---` blob. Compare the ORIGINAL panel
and the RESULT panel row by row; the `[diff]` panel is blank when the design was preserved and busy
when it moved. If you cannot tell what a texture is from `lum`, look at `edge` (structure without
colour) and at the palette line.

## What "good" means
Replace each original (mostly 16..256 px, hand-painted or photo-sourced, DX7-era) with a texture at the
**target size** in `task.json` that is **recognisably the same texture**: same colours, same
composition, same baked-in lighting/grime level, same transparency silhouette and same tiling, but
crisp and detailed at 8x the resolution. It must NOT turn into a modern clean PBR material or a
different design. If the original is a streaky brown road, the result is a crisp streaky brown road.
Textures of one family (one building, one aircraft) must stay consistent with each other.

## First, work out what each texture really IS (this matters most)
Do not trust the file name alone, and do not trust a first glance either. A streaky tan texture named
`gravelroad` looks like wood grain but is a gravel road with wear running along it, and a wood
replacement would be wrong — that mistake once produced a particleboard look because the streak
coherence collapsed (0.11 vs the original's 0.75). Establish the material from ALL of: the `lum`/`edge`
views, the name, the `usage` field in task.json (which world nodes use it, e.g. `someroads`; ground vs
wall; `world_units_per_repeat` = how big one repeat is, which sets the feature scale), the sibling
textures in your task and family, and the original's mean colour. Say in your recipe's comments what
you decided it is and why. Then keep the original's structure: **direction and strength of any streaks
or grain, the feature scale (grain/brick/stone size relative to the tile), and the weathering level.**
The gate measures grain direction/strength (`grain`) and the visible feature scales (`bands`) — a crop
of the wrong material that matches the colour still fails. Use `align_grain(asset, original)` to rotate
an asset's grain to match, `region_stretched(...)` to turn isotropic grit into lengthwise streaks, and a
small `inject_detail_lab` sigma (about target/64 .. target/24) so the original's own mid-scale pattern
survives.

## Inputs (read these first)
- `tasks/<id>/task.json`: per unit: uid, stem, size, target, has_alpha, alpha_is_soft, tileable_guess,
  seam_ratio, look (flat-graphic-likely / photo-painted-likely), distinct_colours_4bit, mean_rgb,
  flipbook_id, `usage`, and `also_named`. `tileable_guess` is only meaningful for opaque surface
  textures: ignore it for decals, sprites and anything with alpha.
- `tasks/<id>/orig/<uid>.png` in code via `orig(uid)`, plus `text_view(uid)` to see it.
- The open-asset library: ~2,200 CC0 colour maps (ambientCG + Poly Haven), 2048 px, indexed in
  `assets/index.jsonl`. Use `lib_search("brick wall", near=tuple(mean_rgb))`, then look at the
  shortlist as text: `print(text_view(lib_image(r["id"]), cols=32))` for each candidate. Compare
  several; do not take the first hit.

## Tools (`tools/texlab.py`, run scripts with `/home/deck/workspace/csvm-hd/venv/bin/python -I script.py`)
Asset ids in examples are only examples: use ids `lib_search` returns.
```python
import sys; sys.path.insert(0, "/home/deck/workspace/csvm-hd/tools")
from texlab import *
TASK = "t0043"; uid = "cliff02__860ab591"
o = orig(uid, "RGB"); t = info(uid)["target"]               # t = [w, h]
base = up_wrap(o, t, wrap=(True, True))                      # tiling-aware enlargement + clean sharpen
cands = lib_search("cliff rock", near=tuple(np.asarray(o).reshape(-1,3).mean(0)), n=18)
det = region(lib_image("Rock030"), t, frac=0.4, seed=1)      # crop of an asset at target size
det = match_color(det, o)                                    # recolour to the original's palette
img = inject_detail_lab(base, det, sigma=t[0]/32, rel=0.15)  # original's layout + asset's fine detail
finish(uid, img, TASK, "up_wrap+libcrop+colormatch+lab-inject", ["Rock030"], fidelity="material")
```
Helpers you should reach for:
- surfaces: `up_wrap`, `region`, `region_stretched(asset,t,frac,seed,stretch=4)` (anisotropic: streaks),
  `make_tileable`, `align_grain`, `match_color`, `inject_detail_lab(base, det, sigma, rel, cfac, mask)`
  (gentle, no hue noise — preferred), `inject_detail` (RGB, stronger), `grain(asset_id, size, amp, seed, vstreak=)`
  (fine multiplicative surface texture, amp 0.02-0.05), `blur`, `quantize`, `seam_ratio`.
- flat art / sprites: `pchip_up` (monotone cubic: no ringing on gradients), `ibp(a, size)`
  (back-projection: reconstruction that shrinks back to the original almost exactly),
  `soft_up(img, size)` (PCHIP colour + back-projected alpha; handles both premultiplied and straight
  alpha automatically — check with `alpha_convention(img)`), `alpha_from_orig(uid, t, cutout=True)` for
  hard cut-outs, `grid_view(uid, path)` + `render_svg` + `svg_try` for hand-drawn vector work.
- looking: `text_view`, `stats_view`, `compare_view`, `palette`.
Write each unit's script to `out/<id>/recipes/<stem>.py` so it can be re-run; keep SVG sources in the
same folder.

## Order of preference (try earlier ones first; ESRGAN-style neural upscaling is a last resort)
1. **Compose from the library** (collage, colour-match, frequency-inject, filters). Best for surfaces:
   ground, rock, brick, concrete, metal plates, wood, roofs, asphalt, plaster, fabric.
2. **Vector, drawn by you**, for flat graphics: logos, letters/glyphs, gauges, signs, cartoon decals,
   nose art, and soft sprites (flares/glows are nested radial gradients + rays; write the SVG by hand in
   the original's pixel units, `viewBox="0 0 W H"`, then `svg_try(uid, svg, path)` and iterate until the
   `compare_view` diff map goes quiet). Fitting a few numeric parameters by least squares (blob centres,
   radii, gradient stops) beats eyeballing: see `out/t0069/recipes/flare_sonic.py`, which reproduced a
   9-blob hand-painted flare to SSIM 0.97 with no images. The automatic tracer `vectorize(img, size)` is
   only a scaffold — it leaves blobby banding.
3. **Hybrid**: smooth-enlarged original as the base plus detail injected from assets.
4. If none of these can reach acceptable quality (typical: painted illustrations with faces/figures,
   tiny unreadable art): `mark(uid, TASK, "esrgan_needed", "why")`. Do not fake it; the maintainer runs
   an upscaler on those. `mark(uid, TASK, "keep_original", "why")` when the original is already the best
   answer (a plain flat fill is better made with `finish` at full size; a 16px icon that is pure noise).

## fidelity (argument to `finish`)
- `"exact"`: the design must be preserved (logos, glyphs, gauges, signs, nose art, text, labels,
  anything with a recognisable picture). The gate shrinks your result and requires it to match.
- `"material"`: only the surface look matters (gravel, brick, rust, wood...). The gate checks colour and
  brightness layout, mean, contrast, grain direction/strength, feature-scale bands, and that an
  originally-seamless tile stays seamless; fine detail is free to change.
Alpha: `finish` copies the original's alpha (smoothly enlarged) — that is usually right. For a hard
cut-out (fences, leaves, cut-out decals) pass `alpha=alpha_from_orig(uid, t, cutout=True)`. If you only
need to *reconstruct* a soft sprite, `soft_up` is the fastest honest answer (see
`out/t0208/recipes/greenhilite.py`, SSIM 1.000) — but do not sharpen its alpha just to satisfy the gate.
Flipbooks (`flipbook_id` set): all frames get the same recipe/asset/seed offsets so the animation still
reads as continuous, and the frame-to-frame motion must survive.

## The gate (`venv/bin/python -I tools/gate.py out/<id>`)
Writes `gate.json` + `compare_<uid>.txt`. Thresholds and what they mean: target size and RGB/RGBA mode;
every library id must exist and be CC0; `exact` → SSIM ≥ 0.78, MAE ≤ 18, silhouette IoU ≥ 0.85;
`material` → low-frequency error ≤ 22, mean shift ≤ 14, contrast ratio 0.35-3, grain angle within 20°
and strength within 0.5-1.8x when the original has grain, feature-scale bands within 0.3-3.2x, seams
preserved; detail must beat a plain Lanczos enlargement by 1.15x **unless** the original is soft/flat
(`orig_flat: true` — glows, gradients, indicator lamps), where matching it exactly *is* the job.
`keep_original` counts as accepted; `esrgan_needed` does not.
A result that passes the gate but looks muddy, wrong-material, off-palette or off-style is not done:
read the `compare_view` panels and judge.

## Loop
1. `text_view` every original; decide a recipe per unit (they differ). 2. Write recipes, run them.
3. Run the gate; **Read `out/<id>/compare_<uid>.txt` for every unit** and fix what the panels show.
4. Up to 3 attempts per unit, then keep the best or `mark(...)`. Be economical with full-res work.

## Rules
- No network. Only the local library. Do not modify anything outside `out/<id>/` (plus scratch files
  inside it). Do not touch `~/Games`. Only python scripts, the gate, and read-only shell commands.
- Every library asset you use must be passed to `finish(... sources=[ids])` (provenance + licence check).
- Deterministic: fixed seeds.
- Some textures are edge crops/flips or downscaled copies of another texture (`*_trans*`, `-64`/`-128`
  suffixes). Where `task.json` gives a unit a `rel` field (`{parent, parent_stem, kind, corner, mae,
  ssim}`) that is proven: upgrade the **parent** once, then derive the child from your parent's HD result
  by cropping the named corner at the same fraction and applying `kind` (see
  `out/t0043/recipes/cliff01_trans1_eada8b02.py`). If a sibling's stem is a suffix of another and there is
  no `rel`, test the hypothesis yourself (compare the child against the parent's four corners) before
  spending attempts on them separately.

## Final reply (under 15 lines — the maintainer aggregates these automatically)
`uid | ok|esrgan_needed|keep_original|failed | gate pass? | method | library ids | self-rating 1-5`.
Then at most one line each for `wants:` (library assets missing), `tooling:` (a helper that would have
saved time) and `notes:` (a surprise about these textures) — only if genuinely useful.
