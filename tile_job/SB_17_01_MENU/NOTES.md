# SB_17_01_MENU notes (tiled FLUX 4x, per-tile prompts)
Prompts: prompts.json (every s4 tile quotes its visible fragments incl. cut words). Assembly: seam_assemble.py (= tools/seam_assemble2.py) + seams_s2/s4.json (seams at min-diff columns/rows, ramp 12; seam rows between text lines, x seams in gaps where possible). Polished with polish_text_tiles.py --n 6 (submitted): 4 words replaced in total.
## Lessons
- Pink script on a black banner gets a glow/halo from FLUX; adding 'printed flat, no glow, no halo' to the prompts reduced it (faint halo remains).
- A banner edge shifted ~2 px between neighbouring tiles; a wide ramp (56) on blank-black banner area hid it at s2.
- Quoting each cut fragment ('Flapjacks & S' (cut by the right edge)) gave exact text in nearly all tiles; kiss print and handwriting 'Come back for more hot stuff, sweetie!' kept.
## Per tile
- s2: 2 attempts (glow fix, seed 7); s4: 1 attempt each, then 6 extra seeds + word stitching.
Imperfect: faint halo around pink script; the s4 tile column x~1670 seam crosses the kiss print (not visible).
