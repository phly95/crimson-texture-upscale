"""mag_finish.py STEM [IN.png OUT.png]: the clean-up of one scrapbook magazine cover after its lettering was restored (see mag_polish.py for the passes): redraw ghosts and invented icon detail
replaced by the original's, colour haze pulled back to the original's low frequencies, smooth alpha. IN defaults to the cover's shipped (approved) up image from the review gallery export."""
import sys
from pathlib import Path
import numpy as np
from PIL import ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mag_polish as mp

SCR = Path("/tmp/claude-1000/-home-deck/8d33977a-9b65-41a7-af0e-5ee75413ed5e/scratchpad")
purple = lambda rgb: (rgb[..., 2].astype(int) > rgb[..., 1].astype(int) + 40) & (rgb[..., 0].astype(int) > rgb[..., 1].astype(int) + 20)


NAME = {"SB_00_00_MAG1": "MAG1", "SB_01_02_MAG2": "MAG2", "SB_11_01_MAG3": "MAG3", "SB_11_01_MAG4": "MAG4", "SB_14_01_MAG5": "MAG5", "SB_21_01_MAG7": "MAG7", "SB_19_01_ZPJCOVER": "ZPJCOVER", "SB_22_01_COMICOVER": "COMICOVER"}


def tmp(stem, n):
    return str(SCR / "finish" / f"{stem}_{n}.png")


def MAG3(src, stem="SB_11_01_MAG3"):
    a = tmp(stem, "a")
    mp.fill_poly(src, a, [(60, 438), (194, 430), (196, 450), (58, 460)], purple)                       # the mosaic of paper blocks under the portrait
    b = tmp(stem, "b")
    mp.ghost_fix(stem, a, b, [(470, 430, 1000, 1050)], sig=6, lo=3, hi=9, amt=1.3, keep_gold=True, bfeather=70)  # motion-blurred plane and crease as the original has them
    c = tmp(stem, "c")
    mp.replace_from_ref(stem, b, c, [(805, 830, 1032, 1240)])                                         # card fan, star, skull: the original's icons
    return c


def MAG4(src, stem="SB_11_01_MAG4"):
    a = tmp(stem, "a")
    mp.ghost_fix(stem, src, a, [(430, 820, 880, 1100)], keep_red=True)                                 # ghost trousers and teal haze around the red caption
    b = tmp(stem, "b")
    mp.replace_from_ref(stem, a, b, [(830, 330, 1000, 640)])                                          # card fan, star, skull: the original's icons
    return b


blue = lambda rgb: (rgb[..., 2].astype(int) > rgb[..., 0].astype(int) + 18) & (rgb[..., 2] > 150)


def MAG2(src, stem="SB_01_02_MAG2"):
    a = tmp(stem, "a")                                                                                  # the redraw turned the U of VOLUME and NUMBER into a D/O shape: both labels are set again
    f = mp.WIN / "SCHLBKB.TTF"
    mp.tilted_text(src, a, "VOLUME LII", 332, 188, 515, 0.103, 23, f, (24, 28, 56), blue)
    b = tmp(stem, "b")
    mp.tilted_text(a, b, "NUMBER TWO", 677, 225, 901, 0.098, 21, f, (24, 28, 56), blue)
    return b


def MAG1(src, stem="SB_00_00_MAG1"):
    a = tmp(stem, "a")                                                                                  # the redraw's script caption reads 'Nathon Zachwy: Gentlemau Pirate!': set again
    mp.tilted_text(src, a, "Nathan Zachary: Gentleman Pirate!", 228, 1599, 1368, -0.118, 58, mp.WIN / "VINERITC.TTF", (238, 229, 204), None, shadow=0.5, rect_up=2.2, rect_down=0.9, thin_th=25)
    b = tmp(stem, "b")
    mp.replace_from_ref(stem, a, b, [(1152, 0, 1235, 95)], feather=6)                                  # the little badge top right: the original's, not invented micro-lettering
    return b


def MAG5(src, stem="SB_14_01_MAG5"):
    a = tmp(stem, "a")                                                                                  # a claw-like blob of the redraw sits where the word 'survive' ends and hides its last letters
    dark = lambda rgb: rgb.astype(int).sum(-1) < 230
    mp.fill_poly(src, a, [(472, 978), (520, 972), (524, 1024), (470, 1024)], dark, sig=(3, 8, 20))
    b = tmp(stem, "b")
    f = mp.WIN / "BELL.TTF"
    w = ImageFont.truetype(str(f), 21.6).getlength("ive")
    mp.tilted_text(a, b, "ive", 469, 1001, 469 + w * np.cos(np.arctan(-0.07)), -0.07, 15, f, (222, 222, 220), lambda r: np.ones(r.shape[:2], bool), wipe=False, soft=1.2, stroke=1)
    return b


def MAG7(src, stem="SB_21_01_MAG7"):
    a = tmp(stem, "a")                                                                                  # the faint white caption over the buildings has a ghost copy of its letters: set again
    mp.tilted_text(src, a, "POLICE HARASS CITIZENS!", 226, 763, 580, -0.124, 20, mp.WIN / "arialbd.ttf", (234, 232, 226), None, rect_up=1.2, rect_down=0.5, thin_th=22, soft=0.8)
    return a


FIX = {"SB_14_01_MAG5": MAG5, "SB_21_01_MAG7": MAG7, "SB_11_01_MAG3": MAG3, "SB_11_01_MAG4": MAG4, "SB_01_02_MAG2": MAG2, "SB_00_00_MAG1": MAG1}
POLISH = {"SB_22_01_COMICOVER": (10, 0)}                                                                   # flat comic art: colour only: the mint glow around the lettering goes, the redraw's shading stays


# ---- pass 2 (visual QA): applied on top of the pass-1 result, i.e. on SCR/gallery/<NAME>_hd.png (the shipped custom version) ----
import cv2                                                                                               # noqa: E402
from PIL import Image                                                                                    # noqa: E402


def t2(stem, n):
    return str(SCR / "finish" / f"{stem}_p2{n}.png")


def _arr(p):
    return np.asarray(Image.open(p))


def MAG4_2(src, stem="SB_11_01_MAG4"):
    import mag_sb_11_01_mag4 as ms
    a = t2(stem, "a")                                                    # right edge: the redraw drew a black band and a bright paper-edge strip with specks; the original has a soft shadow
    band = mp.edge_band(_arr(src)[..., 3], 60, [[(840, 540), (1028, 540), (1028, 1268), (840, 1268)]]) | mp.edge_band(_arr(src)[..., 3], 45, [[(960, 350), (1028, 350), (1028, 540), (960, 540)]])
    mp.ref_in_mask(stem, src, a, band, feather=8)
    Hu, Wu = mp.upright_shape(stem)                                      # red caption: legs, boots and floor under it from the original (the redraw's are hazy, ghosted, with
    f = ImageFont.truetype(ms.IMPACT, 100); f = ImageFont.truetype(ms.IMPACT, 100 * ms.CAP_W / f.getlength(ms.CAPTION[0]))   # white slats), the caption set again crisp
    def draw(d):
        for k, t in enumerate(ms.CAPTION):
            d.text((ms.CAP_X, ms.CAP_BASE + k * ms.CAP_PITCH), t, font=f, fill=255, anchor="ls")
    M = cv2.getRotationMatrix2D(ms.CAP_C, ms.ANG, 1.0)
    cap = mp.page_layer(stem, cv2.warpAffine(ms.layer((Hu, Wu), draw), M, (Wu, Hu), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR))
    rc = mp.ref_clean(stem, red_grow=10)
    o = mp.sharp(rc, 2.0, 1.1) + np.random.default_rng(2).normal(0, 1.5, rc.shape[:2])[..., None]
    box = np.maximum(mp.box_mask(cap.shape, [(450, 840, 960, 1300)], 30),
                     cv2.GaussianBlur(cv2.dilate((cap > 0.05).astype(np.uint8), np.ones((41, 41), np.uint8)).astype(np.float32), (0, 0), 8))
    b = t2(stem, "b"); mp.blend_rgb(a, b, o, box)
    c = t2(stem, "c"); mp.wipe_and_set(b, c, np.zeros(cap.shape, bool), cap, (205, 52, 42))
    d = t2(stem, "d")                                                    # dim copies of the old headline letters under each line
    prot = mp.text_mask(_arr(c)[..., :3], th=40, open_k=15, minL=150, grow=2)
    mp.clean_bg(c, d, [[(125, 245), (560, 250), (560, 470), (125, 465)]], prot, dev=8, thin=13, median=21)
    e = t2(stem, "e")                                                    # the translucent (double-exposed) forearms: brightness of the original's opaque sleeves, the redraw's folds
    arr = _arr(d)
    m = cv2.GaussianBlur(mp.poly_mask(arr.shape, [[(492, 545), (575, 540), (580, 650), (530, 658), (495, 640)],
                                                  [(745, 610), (850, 600), (870, 640), (845, 705), (780, 712), (745, 690)]]).astype(np.float32), (0, 0), 6)
    Image.fromarray(np.dstack([mp.lock(arr[..., :3], mp.ref_page(stem, 4), 6, 4, m), arr[..., 3]])).save(e)
    g = t2(stem, "g")                                                    # invented white crackle along the dark left margin
    arr = _arr(e); rgb = arr[..., :3]
    band = mp.edge_band(arr[..., 3], 22, [[(0, 330), (200, 330), (200, 1268), (0, 1268)]])
    mp.thin_out(e, g, band, k=9, th=18)
    h = t2(stem, "h")                                                    # the floor left of the boots: the original's light floor (the redraw's is a grey haze)
    mp.ref_in_mask(stem, g, h, mp.poly_mask(arr.shape, [[(345, 1105), (470, 1000), (470, 1268), (250, 1268), (250, 1112)]]), feather=12)
    return h


def MAG3_2(src, stem="SB_11_01_MAG3"):
    import mag_sb_11_01_mag3 as ms
    a = t2(stem, "a")                                                    # red caption: soft, with a notch after 'was': wiped (red ink and its pink fringe) and set again
    cap = mp.set_lines(stem, ms.CAPTION, ms.IMPACT, ms.CAP_X, ms.CAP_BASE, ms.CAP_PITCH, w0=ms.CAP_W)
    rgb = _arr(src)[..., :3].astype(int)
    zone = cv2.dilate((cap > 0.05).astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    red = ((rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2])) > 14) & (rgb[..., 0] > 35) & zone
    wm = (cv2.dilate(red.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) | (cv2.dilate((cap > 0.02).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    mp.wipe_and_set(src, a, wm, cap, ms.CAP_INK, sig=(3, 8, 20, 50))
    b = t2(stem, "b")                                                    # bottom-left purple: the redraw's white crease lines and a dark haze; the original's purple (red caption painted out)
    arr = _arr(a)
    bg = mp.ref_clean(stem, red_grow=10)
    o = mp.sharp(bg, 1.6, 0.8) + np.random.default_rng(2).normal(0, 1.5, bg.shape[:2])[..., None]
    m = cv2.GaussianBlur(mp.poly_mask(arr.shape, [[(40, 1012), (260, 1000), (345, 1060), (345, 1240), (40, 1240)]]).astype(np.float32), (0, 0), 10)
    prot = mp.text_mask(arr[..., :3], th=40, open_k=15, minL=140, grow=3); prot[1040:] = False
    prot |= cv2.dilate((cap > 0.02).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    mp.blend_rgb(a, b, o, m * (1 - cv2.GaussianBlur(prot.astype(np.float32), (0, 0), 1.5)))
    c = t2(stem, "c")                                                    # grey bloom around the column's lines on the black: the original's black under the crisp letters
    arr = _arr(b); rgb = arr[..., :3]
    region = mp.poly_mask(arr.shape, [[(150, 560), (470, 520), (560, 600), (560, 1050), (150, 1080)]])
    bgn = mp.ref_notext(stem, region=cv2.dilate(region.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0, th=12, grow=6)
    dark = (cv2.GaussianBlur(cv2.cvtColor(bgn, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 4) < 48) & region
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ink = np.clip((L - 70) / 60, 0, 1) * cv2.dilate(mp.text_mask(rgb, th=35, open_k=13, minL=110, grow=1).astype(np.uint8), np.ones((3, 3), np.uint8))
    o = bgn.astype(np.float32) + np.random.default_rng(3).normal(0, 1.2, L.shape)[..., None]
    w = (cv2.GaussianBlur(dark.astype(np.float32), (0, 0), 6) * (1 - ink))[..., None]
    Image.fromarray(np.dstack([(rgb * (1 - w) + o * w).clip(0, 255).astype(np.uint8), arr[..., 3]])).save(c)
    d = t2(stem, "d")                                                    # the black disc's edge: the redraw made it stepped and bitten; redrawn as the circle the original has
    C = (768.64, 872.71, 619.98)                                         # (fitted to the edge, within 2.5 px everywhere)
    rgb = _arr(c)[..., :3]
    pur = lambda ai: (ai[..., 2] > ai[..., 1] + 40) & (ai[..., 0] > ai[..., 1] + 20) & (ai.sum(-1) > 330)
    blk = lambda ai: ai.sum(-1) < 120
    col = mp.column_layer(stem, ms.COLUMN, ms.PARA_START, ms.COL_X, ms.COL_INDENT, ms.COL_Y0, ms.COL_PITCH, ms.COL_SIZE, ms.BELL, ms.BELLI)
    yy, xx = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]]
    dd = np.hypot(xx - C[0], yy - C[1]) - C[2]
    zone = (np.abs(dd) < 24) & (yy > 440) & (yy < 1100)
    def fix(pi, po):                                                     # under the column the letters are lifted off (their exact layer), the edge redrawn, the letters laid back
        bb = _arr(pi)[..., :3].astype(int)
        mp.redo_circle(pi, po, *C, pur(bb) & (bb[..., 1] < 112), blk(bb), half=24, region=zone)
    c2 = t2(stem, "c2"); mp.relay_text(c, c2, col, ms.COL_INK, zone, fix=fix)
    prot = mp.gold_mask(rgb, 1) > 0
    bb = _arr(c2)[..., :3].astype(int)
    mp.redo_circle(c2, d, *C, pur(bb), blk(bb), half=10, protect=prot, region=mp.poly_mask(rgb.shape, [[(100, 220), (940, 220), (940, 300), (330, 440), (100, 440)]]))
    e = t2(stem, "e")                                                    # stray dots above 'Air' and 'P', ghost specks after 'Audiences!'
    pp = lambda r: (r[..., 2].astype(int) > r[..., 1].astype(int) + 40)
    dk = lambda r: r.astype(int).sum(-1) < 110
    mp.fill_poly(d, e, [(400, 327), (416, 327), (416, 343), (400, 343)], pp, sig=(3, 8, 20))
    for poly, paper in (([(492, 292), (508, 292), (508, 308), (492, 308)], pp), ([(828, 392), (878, 392), (878, 425), (828, 425)], dk),
                        ([(838, 389), (850, 389), (850, 399), (838, 399)], dk), ([(868, 396), (880, 396), (880, 412), (868, 412)], dk)):
        mp.fill_poly(e, e, poly, paper, sig=(3, 8, 20))
    f = t2(stem, "f")                                                    # white scratches on the left margin, a pink blot on the edge
    arr = _arr(e)
    mp.thin_out(e, f, mp.poly_mask(arr.shape, [[(0, 432), (75, 432), (75, 478), (0, 478)]]), k=9, th=14)
    mp.fill_poly(f, f, [(30, 540), (58, 540), (58, 604), (30, 604)], lambda r: pur(r.astype(int)) & (r[..., 1] < 112), sig=(3, 8, 20))
    mp.thin_out(f, f, mp.poly_mask(arr.shape, [[(10, 300), (80, 300), (80, 362), (10, 362)], [(10, 238), (130, 238), (130, 266), (10, 266)]]), k=9, th=14)
    return f


def MAG1_2(src, stem="SB_00_00_MAG1"):
    import mag_sb_00_00_mag1 as ms
    def band(up):                                                        # in the straightened frame the mag_sb script measured: the imprint sat on the dark rule (a dotted
        H, Wd = up.shape[:2]; out = up.copy(); rng = np.random.default_rng(31)   # red line through its baseline, a darker wipe box), the black strip had flat boxes, specks, a scratch
        R, G = up[..., 0], up[..., 1]
        reg = np.zeros((H, Wd), bool); reg[603:666, 450:1452] = True       # the black strip: the original's black (13, 6, 18) + grain, down to the grey photo paper
        red = (R - G > 30) & reg
        paperg = (up.sum(-1) > 150) & ~red & (np.abs(up[..., 0] - up[..., 2]) < 30)
        paperg = cv2.morphologyEx(paperg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)) > 0; paperg[:615] = False
        reg &= ~(np.cumsum(paperg.astype(np.int32), 0) > 0)
        reg = cv2.erode(cv2.dilate(reg.astype(np.uint8), np.ones((3, 3), np.uint8)), np.ones((3, 3), np.uint8)) > 0
        g = cv2.GaussianBlur(rng.normal(0, 3.0, (H, Wd)).astype(np.float32), (0, 0), 0.7)[..., None]
        out = np.where(reg[..., None], np.array((13, 6, 18), np.float32) + g, out)
        mr = reg.astype(np.float32)
        rb = np.zeros((H, Wd), bool); rb[564:594, 604:1060] = True        # the imprint's red: refilled from the red round it, a clean rule at rows 589.5..592.5
        redp = (R > 150) & (G < 60) & (up[..., 2] < 70)
        keep = redp & ~rb; keep[:540] = False; keep[590:] = False
        rf = mp._fill(up, keep, rb, (3, 8, 24, 80)) + cv2.GaussianBlur(rng.normal(0, 4.0, (H, Wd)).astype(np.float32), (0, 0), 0.7)[..., None]
        yy = np.arange(H, dtype=np.float32)[:, None]
        rule = np.broadcast_to(np.clip(np.minimum(yy - 589.0, 593.0 - yy) + 0.5, 0, 1), (H, Wd))[..., None]
        out = np.where(rb[..., None], rf * (1 - rule) + np.array((42, 9, 6), np.float32) * rule, out)
        mr = np.maximum(mr, rb.astype(np.float32))
        for text, font, cap, a0, a1, bl, ink in [("ES, DC, PC 45 cents – ME, TX, OK 35 cents", ms.F_COND, 18, 456, 868, 626.5, ms.RED_PRICE),
                                                 ("March 17, 1939", ms.F_COND, 30, 1178, 1437, 638.5, ms.RED_DATE),
                                                 ("Reg. Canadian Patent Office – Printed in Canada", ms.F_COND, 10, 620, 1028, 587.0, ms.IMPRINT_INK)]:   # 2.5 px higher: clear of the rule
            a = ms.text_mask(text, font, cap, a0, a1, bl, (H, Wd))
            out = out * (1 - a[..., None]) + np.asarray(ink, np.float32) * a[..., None]
        return out, cv2.GaussianBlur(mr, (0, 0), 1.5)
    a = t2(stem, "a"); mp.upright_edit(stem, src, a, band)
    b = t2(stem, "b")                                                    # the embossed outlines of the redraw's old script letters round the new caption
    rgb = _arr(a)[..., :3].astype(int)
    cream = (rgb.sum(-1) > 540) & (rgb[..., 0] - rgb[..., 2] > 10) & (rgb[..., 0] - rgb[..., 2] < 70)
    prot = cv2.dilate(cream.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    mp.clean_bg(a, b, [[(210, 1520), (600, 1475), (612, 1600), (210, 1640)], [(1150, 1400), (1400, 1360), (1400, 1520), (1150, 1540)],
                       [(870, 1470), (1150, 1430), (1150, 1560), (870, 1590)]], prot, dev=7, thin=13, median=25)
    return b


def MAG5_2(src, stem="SB_14_01_MAG5"):
    import mag_sb_14_01_mag5 as ms
    from PIL import ImageDraw
    a = t2(stem, "a")                                                    # orange headline: the redraw added inline outlines and a 3-D extrusion (a doubled look): only its orange
    meas = lambda o: np.clip((o[..., 0] - o[..., 2] - 40) / 140.0, 0, 1)   # where the original is orange too is kept, as one flat ink; the rest is refilled from the photo
    mp.colour_text_gate(stem, src, a, [(70, 370), (830, 320), (860, 560), (70, 610)], meas, (250, 130, 8), gate_thr=0.2, gate_grow=3, wipe_th=0.08)
    b = t2(stem, "b")                                                    # right end of the column: a pale invented panel and blotches behind the lines; the original's backdrop
    arr = _arr(a); reg = mp.poly_mask(arr.shape, [[(440, 860), (575, 850), (575, 1070), (440, 1075)]])
    bg = mp.ref_notext(stem, region=cv2.dilate(reg.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0, th=14, grow=5)
    mp.blend_rgb(a, b, mp.sharp(bg, 2.0, 0.8) + np.random.default_rng(2).normal(0, 1.5, bg.shape[:2])[..., None], cv2.GaussianBlur(reg.astype(np.float32), (0, 0), 8))
    # the column as the print wraps it (read from the original at 1x, upright frame): the first paragraph's measure is narrower than the second's, and the print
    # ends the second paragraph at 'obligated' (the string's 'directors, writers, and actors.' is not printed); every line start and end was checked against the original
    LINES = ["Film star Lana Cooper today announced", "that she is opening a new studio with the aid", "of private investors. The new studio, Pirate",
             "Pictures, is intended to be a small, actor-", "friendly concern, which aims to release high-", "quality entertainment with a minimum of",
             "\"executive supervision,\" and no long term", "contracts.", "No details are available yet, but Cooper is",
             "slated to star in and direct the first film. Rumors", "are already spreading that several stars near", "the end of their contracts have contacted Cooper",
             "for roles. Established studios issued statements", "supporting Miss Cooper's new endeavor, although", "privately most claim that no studio can survive",
             "without a stable of contractually obligated", "(Continued on p. 5)"]
    def layers(H, Wd):
        f = ImageFont.truetype(ms.BELL, ms.COL_SIZE)
        old = Image.new("L", (Wd, H), 0); do = ImageDraw.Draw(old)
        for t, x, yb in ms.column_lines(ms.column_paragraphs()):          # where the pass-1 lines are (to be wiped)
            do.text((x, yb), t, font=f, fill=255, anchor="ls", stroke_width=ms.COL_STROKE, stroke_fill=255)
        new = Image.new("L", (Wd, H), 0); dn = ImageDraw.Draw(new)
        for i, t in enumerate(LINES):
            dn.text((ms.COL_X + (ms.COL_INDENT if i in (0, 8) else 0), ms.COL_BASE + ms.COL_PITCH * i), t, font=f, fill=255, anchor="ls",
                    stroke_width=ms.COL_STROKE, stroke_fill=255)
        return np.asarray(old, np.float32) / 255, np.asarray(new, np.float32) / 255
    def col(up):                                                         # the column: every light stroke on the dark backdrop (the pass-1 lines, their bloom, ghosts) wiped,
        H, Wd = up.shape[:2]                                             # the print's lines laid with less blur
        L1 = layers(H, Wd)[1]
        g = cv2.cvtColor(up.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        bg = cv2.morphologyEx(g, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
        zone = np.zeros((H, Wd), bool); zone[730:1200, 100:620] = True
        zone &= cv2.GaussianBlur(bg, (0, 0), 6) < 95                      # the dark backdrop only (the blouse and the ring stay)
        wm = (((g - bg) > 14) & zone) | ((cv2.GaussianBlur(L1, (0, 0), 1.5) > 0.03) & zone)
        wm = cv2.dilate(wm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        keep = ~wm & (g < 95)
        fill = mp._fill(up, keep, wm, (3, 6, 12, 30))
        f8 = fill.clip(0, 255).astype(np.uint8)
        fill = np.dstack([cv2.medianBlur(f8[..., c], 7) for c in range(3)]).astype(np.float32)
        out = np.where(wm[..., None], fill + np.random.default_rng(41).normal(0, 1.5, (H, Wd))[..., None], up)
        ref = mp.to_upright(stem, mp.ref_notext(stem, th=12, grow=5)).astype(np.float32)   # the arm under the top lines' right ends: the original's, its blurred print painted out
        gr = cv2.cvtColor(ref.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        dref = cv2.GaussianBlur((cv2.GaussianBlur(cv2.morphologyEx(gr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))), (0, 0), 4) < 80).astype(np.float32), (0, 0), 1.5)
        arm = np.zeros((H, Wd), np.float32); arm[700:835, 440:640] = 1
        arm = cv2.GaussianBlur(arm * (1 - dref), (0, 0), 3)               # the arm's patch (the redraw's is blotchy, with the old lines' ghosts): the original's
        out = out * (1 - arm[..., None]) + mp.sharp(ref, 2.0, 0.8) * arm[..., None]
        al = np.clip(cv2.GaussianBlur(L1, (0, 0), 0.7) * 1.1, 0, 1)[..., None]
        out = out * (1 - al) + np.array(ms.COL_INK, np.float32) * al
        m = cv2.GaussianBlur(cv2.dilate((wm | (L1 > 0.02)).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 1.5)
        return out, np.maximum(m, arm)
    c = t2(stem, "c"); mp.upright_edit(stem, b, c, col)
    d = t2(stem, "d")                                                    # invented white cracks along the left margin (the column's letters are kept out of it)
    Hu, Wu = mp.upright_shape(stem)
    txt = mp.page_layer(stem, cv2.dilate((layers(Hu, Wu)[1] > 0.02).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(np.float32)) > 0.02
    mp.thin_out(c, d, mp.poly_mask(arr.shape, [[(0, 60), (75, 60), (75, 1264), (0, 1264)]]) & ~txt, k=9, th=16)
    return d


def MAG7_2(src, stem="SB_21_01_MAG7"):
    import mag_sb_21_01_mag7 as ms
    from PIL import ImageDraw
    a = t2(stem, "a")                                                    # the redraw drifted the top of the cover ~15 px up (to 0 at mid-page): moved back onto the original's geometry
    mp.align_to_ref(stem, src, a, sigma=100)
    b = t2(stem, "b")                                                    # under the railing the redraw drew a hard-slatted reflection and blotches: the original's soft streak
    arr = _arr(a)
    reg = mp.poly_mask(arr.shape, [[(240, 875), (520, 845), (560, 870), (620, 870), (660, 935), (240, 945)]])
    o = mp.sharp(mp.ref_page(stem, 4), 1.6, 0.6) + np.random.default_rng(2).normal(0, 1.5, arr.shape[:2])[..., None]
    mp.blend_rgb(a, b, o, cv2.GaussianBlur(reg.astype(np.float32), (0, 0), 6))
    def col(up):                                                         # the body column: ghost strokes of the old type under and between the lines and the haze patch are wiped
        H, Wd = up.shape[:2]                                             # (everything darker than the paper, plus the lines' own footprint), the lines set again crisper
        f = ImageFont.truetype(ms.BELL, ms.BODY_SIZE)
        lay = Image.new("L", (Wd, H), 0); dr = ImageDraw.Draw(lay)
        for t, x, yb in ms.wrap_lines():
            dr.text((x, yb), t, font=f, fill=255, anchor="ls")
        L0 = np.asarray(lay, np.float32) / 255
        zone = cv2.dilate((L0 > 0.02).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
        x0, y0, x1, y1 = ms.BODY_BOX; zone[y0:y1, x0:x1] = True
        zone = cv2.erode(cv2.dilate(zone.astype(np.uint8), np.ones((15, 15), np.uint8)), np.ones((15, 15), np.uint8)) > 0
        g = cv2.cvtColor(up.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        paper = cv2.GaussianBlur(cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))), (0, 0), 3)
        dark = ((g < paper - 12) | (cv2.GaussianBlur(L0, (0, 0), 1.2) > 0.03)) & zone
        wm = cv2.dilate(dark.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        out = np.where(wm[..., None], mp._fill(up, ~wm, wm, (2, 4, 8, 20, 50)), up)
        o8 = out.clip(0, 255).astype(np.uint8)
        med = np.dstack([cv2.medianBlur(o8[..., c], 9) for c in range(3)]).astype(np.float32)
        out = np.where(wm[..., None], med, out) + np.random.default_rng(43).normal(0, 1.6, (H, Wd))[..., None] * wm[..., None]
        al = np.clip(cv2.GaussianBlur(L0, (0, 0), 0.7) * 1.1, 0, 1)[..., None]
        return out * (1 - al) + np.array((58, 52, 46), np.float32) * al, cv2.GaussianBlur(zone.astype(np.float32), (0, 0), 2)
    c = t2(stem, "c"); mp.upright_edit(stem, b, c, col)
    d = t2(stem, "d")                                                    # the skyline blocks behind the price and the target emblem (ghost rings, grey disc shadow): the original's
    mp.ghost_fix(stem, c, d, [(40, 100, 235, 230), (80, 232, 320, 440), (320, 296, 1000, 340)], sig=5, lo=5, hi=12, bfeather=12, amt=1.3)
    mp.thin_out(d, d, mp.poly_mask(arr.shape, [[(0, 100), (70, 100), (70, 1328), (0, 1328)]]), k=9, th=16)   # invented cracks on the left margin
    paper = lambda r: (r.astype(int).sum(-1) > 560) & ((r[..., 0].astype(int) - r[..., 1].astype(int)) < 10)
    e = t2(stem, "e"); mp.fill_poly(d, e, mp.band_poly(270, 660, 332, 275, 8), paper, sig=(4, 10, 30))       # a pink ghost copy of the red line under the slogan
    mp.fill_poly(e, e, [(262, 272), (312, 268), (312, 298), (262, 304)], paper, sig=(4, 10, 30))                # a faint ghost 'TH' left of the slogan, and its trail
    mp.fill_poly(e, e, [(268, 312), (335, 304), (335, 322), (268, 330)], paper, sig=(4, 10, 30))                # below the red line (both left by the realignment)
    g = t2(stem, "g")                                                    # teal cast round the logo: the original's colour (brightness kept)
    arr = _arr(e)
    Image.fromarray(np.dstack([mp.lock(arr[..., :3], mp.ref_page(stem, 4), 6, 0, mp.box_mask(arr.shape, [(0, 0, 1124, 250)], 30)), arr[..., 3]])).save(g)
    return g


def ZPJCOVER_2(src, stem="SB_19_01_ZPJCOVER"):
    a = t2(stem, "a")                                                    # invented paper cracks over the Z, the white strip above the title and the right edge: long thin
    arr = _arr(src); rgb = arr[..., :3]                                  # (7 px) light or dark lines there are replaced by their local median
    reg = mp.poly_mask(arr.shape, [[(140, 40), (430, 40), (430, 195), (300, 215), (300, 330), (140, 330)], [(140, 40), (1000, 40), (1000, 105), (140, 105)],
                                   [(1170, 140), (1264, 140), (1264, 600), (1170, 600)]])
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thin = (((cv2.morphologyEx(L, cv2.MORPH_CLOSE, se) - L) > 10) | ((L - cv2.morphologyEx(L, cv2.MORPH_OPEN, se)) > 10)) & reg
    n, lab, st, _ = cv2.connectedComponentsWithStats(thin.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = np.hypot(st[1:, 2], st[1:, 3]) > 25
    m = cv2.dilate(keep[lab].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    fi = rgb.astype(int)                                                 # the gold date on the black corner strip is never touched (a crack crossing it stays)
    gold = (fi[..., 0] > 110) & (fi[..., 0] - fi[..., 2] > 50) & (fi[..., 1] > 80) & mp.poly_mask(arr.shape, [[(1000, 40), (1264, 40), (1264, 420), (1000, 420)]])
    m &= ~(cv2.dilate(gold.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    med = np.dstack([cv2.medianBlur(rgb[..., c], 11) for c in range(3)]).astype(np.float32)
    w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)[..., None]
    Image.fromarray(np.dstack([(rgb * (1 - w) + med * w).clip(0, 255).astype(np.uint8), arr[..., 3]])).save(a)
    return a


def COMICOVER_2(src, stem="SB_22_01_COMICOVER"):
    import mag_comic_fix                                                 # title flattened (no bevel, glow, drips), 'Fortune Hunter!' set again, title shading and statue vignettes
    return mag_comic_fix.run(src, str(SCR / "finish" / f"{stem}_p2"))


FIX2 = {"SB_11_01_MAG4": MAG4_2, "SB_11_01_MAG3": MAG3_2, "SB_00_00_MAG1": MAG1_2, "SB_14_01_MAG5": MAG5_2, "SB_21_01_MAG7": MAG7_2, "SB_19_01_ZPJCOVER": ZPJCOVER_2, "SB_22_01_COMICOVER": COMICOVER_2}


def main():
    """mag_finish.py STEM [IN OUT]: pass 2 on SCR/gallery/<NAME>_hd.png (default).  mag_finish.py --pass1 STEM IN [OUT]: the original pass 1 (lettering fixes + lock + smooth alpha)."""
    a = sys.argv[1:]
    (SCR / "finish").mkdir(exist_ok=True)
    if a[0] == "--pass1":
        stem = a[1]; src = a[2]; out = a[3] if len(a) > 3 else tmp(stem, "final")
        if stem in FIX:
            src = FIX[stem](src)
        sc, sl = POLISH.get(stem, (28, 24))
        print(mp.polish(stem, src, out, sc, sl))
        return
    stem = a[0]
    src = a[1] if len(a) > 1 else str(SCR / 'gallery' / (NAME[stem] + '_hd.png'))
    out = a[2] if len(a) > 2 else tmp(stem, "final")
    res = FIX2[stem](src) if stem in FIX2 else src
    im = np.asarray(Image.open(res).convert("RGBA"))
    a0 = np.asarray(Image.open(src).convert("RGBA"))[..., 3]
    Image.fromarray(np.dstack([im[..., :3], a0])).save(out)                 # alpha: exactly the input's
    print(out)


if __name__ == "__main__":
    main()
