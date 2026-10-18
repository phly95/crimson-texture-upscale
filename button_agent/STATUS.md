# Button agent status (pass 2, shared master plate)

**9 / 59 SUBMITTED, 50 BLOCKED** (47 on the colour gate, 3 are a different asset). All 56 lettered buttons are rendered
(`sprites/<name>/draw.png`, recipes in `recipes/`). Overview of all 59: `out/_overview.png`, with green = submitted, amber =
rendered but blocked, red = not drawn.

## Submitted (all gates pass; contact sheets inspected)
| button | de_mean | button | de_mean |
|---|---|---|---|
| PF_B_VIDEO | 7.74 | EM_B_SAVEGAME | 9.33 |
| PF_B_AUDIO | 7.79 | EM_B_LOADGAME | 9.63 |
| CM_B_CANCEL | 8.74 | PF_B_MAINMENU | 9.85 |
| PF_B_CONTROLS | 8.76 | MM_B_CAMPAIGN | 9.90 |
| EM_B_RESUME | 8.79 | | |

### PF_B_CONTROLS: the five review points
1. The dark band across the top rim is gone. That was a hard-coded "rim alignment" loop that painted flat rows; the rim is now the master's own.
2. The shadow is soft: blurred (σ 1.1 px orig), small offset (0.9, 1.6), with a dense core under the letters as in the originals,
   plus a light warm glow in state 2. No stair-stepped block.
3. Orange and red have a vertical gradient (light top, orange to red / deep red bottom), a bevel (lighter top edges, darker
   bottom edges), and an amber or red glow tinting the dark ring round the letters.
4. State 1's whole parchment panel is recoloured evenly by one Lab mean/std transfer measured over all originals. The frame and
   filigree are untouched.
5. The letters are embossed (bevel, tight dark contour, soft shadow, halo) and stay crisp: vector glyphs at 3x supersampling.

## BLOCKED: colour gate (de_mean ≥ 10), 47 buttons
These render correctly and consistently (see the overview), but cannot reach de_mean < 10 with the agreed design: one shared
FLUX plate and one modern typeface.

de_mean per button:
- 10-11.5: PF_B_CANCELCHANGES 10.46, PF_B_GAMEOPTIONS 10.67, PF_B_ACCEPTCHANGES 10.68, LS_B_DELETEGAME 10.82,
  CM_B_DELETEPLAYER 10.86, MM_B_QUIT 11.05, KB_B_RESETTODEFAULT 11.09, CR_B_ABOUT 11.23
- 11.5-13: MS_B_DONE 11.76, PX_B_SELLPLANES 12.40, CM_B_START 12.42, LS_B_SAVE 12.69, MM_B_CREDITS 12.70, LS_B_LOAD 12.81,
  PC_B_NEXTMISSION 12.88
- 13-14: GN_B_CONTINUE 13.11, PS_B_BUILDNEWPLANE 13.47, PX_B_EXPORTDESIGN 13.50, PX_B_READYTOPURCHASE 13.69,
  PX_B_READYTOEXPORT 13.79, GN_B_RETURNTOCABIN 13.83, SB_B_EXPORTTODESKTOP 13.84, PX_B_CANCELEXPORT 13.85,
  MM_B_PREFERENCES 13.91, OL_B_ACCEPTLOADOUT 13.97
- 14-15: GN_B_RETURNTOGAME 14.07, PC_B_SAVEGAME 14.14, PS_B_ACCEPTSELECTIONS 14.26, MM_B_INSTANTACTION 14.27,
  IA_B_FLYNEWMISSION 14.30, IA_B_EXITINSTANTACTION 14.31, FC_B_RETURNTOBRIEFING 14.59, PS_B_CANCELSELECTIONS 14.63,
  FC_B_FLYMISSION 14.63, MP_B_EXITMULTIPLAYER 14.77, OL_B_CANCELLOADOUT 14.86
- 15-16.8: PC_B_PLANECONSTRUCTION 15.02, SB_B_VIEWALLMISSIONS 15.06, MM_B_MULTIPLAYER 15.18, PC_B_CHANGEMEMENTO 15.30,
  GN_B_BUILDCUSTOMPLANE 15.31, MS_B_ACCEPTCHANGES 15.80, PC_B_RETURNMAINMENU 15.90, PX_B_CANCELPURCHASE 15.95,
  MS_B_CANCELCHANGES 16.04, PC_B_PREVIOUSMISSIONS 16.34, MS_B_RETURNTOSCRAPBOOK 16.75

All other gates pass for these 47: IoU 0.95-0.998, alpha, rim and seam.

**Why the gate can't be met.** de_mean is a per-pixel comparison against the original at 1x. Breakdown, for example
PC_B_RETURNMAINMENU:
- The plate away from the text is at dE ~13-15 even though its Lab mean/std match. The master's mottling is a different pattern
  from each original's (the original panels have L std ~11 at 1x), and two uncorrelated textures cannot get closer pixel by pixel.
  PF_B_CONTROLS-size buttons pass because the master was painted from that original.
- The text zone is at dE 20-27: at 11-12 px cap height, a 1 px stem offset between two typefaces is enough.

**Tried, with the effect on de_mean:**
- Per-class frame weights and rim fitting: −3 to −4 on the small class. This is in the renderer.
- Carrying each original's own low-frequency mottling onto the plate: −0.5 to −1. In the renderer.
- Text-box calibration and glyph alignment to the original letter centres: −0.5. In the renderer.
- Seven different condensed fonts: ±0.3.
- A stronger mottling transfer: −0.3, but it makes visible blurry smudges, so it was rejected.

Reaching < 10 on the 32/34 px class would take the original's own pixel texture and letterforms, which is an upscale, not the
shared-plate redraw that was agreed.

**Owner decision needed:** accept these on visual review (e.g. a family-specific de_max of about 17 for the buttons group; the
silhouette, rim and alpha gates still apply), or change the design. They are ready: `recipes/<name>.py`, then
`sprite_tool.py submit <name>` once the gate allows it.

## BLOCKED: different asset (not drawn), 3 buttons
IA_B_VIEWSTORY, IA_B_WEAPONLOADOUTFLYMISSION, PX_B_PURCHASENOW are blank paper plates: no lettering, no steel frame, with
white / beige / cream / dark-brown states. They cannot be built from master_plate.png and need their own design decision.
Their recipes raise SystemExit with this note.

## Doubts / notes
- **Mottling transfer**: it makes each plate carry its original's blotch layout. All plates still share the master's grain, frame
  and colour, and the plate mean/std agree within about 1 L* (consistency.py), but the plates are no longer pixel-identical.
  Without it, MM_B_CAMPAIGN, PF_B_MAINMENU and EM_B_LOADGAME would fall above 10.
- **MM_ 42 px family**: the originals draw a separate thin dark line under each plate and put the bar higher. The renderer does
  not reproduce that line, which is why its IoU is ~0.95 instead of 0.99.
- **EM_B_LOADGAME / SAVEGAME**: the original sheets step each plate up by about 2 px per state (42 px plates in 44 px slots);
  the render reproduces this, including the next plate's rim showing at the bottom of each slot.
- **consistency.py**: flags EM_'s plate statistics because of its fixed panel window on stepped plates (a measurement artefact,
  checked by eye). It also flags slightly brighter letter cores on the largest caps, where the bevel is a smaller share of the core.
