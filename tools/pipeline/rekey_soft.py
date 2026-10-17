"""rekey_soft.py STEM [--neutral] [--gmax X] [--no-bloom] [--install] : soft re-key of a unit's keyed RGBA versions (first pass + retry, every seed).
soft green key on the colour-matched redraw (greenkey.green_key; --neutral = neutralise rim colours, for white/grey subjects), then bloom_alpha.bloom (original's soft alpha + glow colour; --gmax replaces olive pixels), then green capped below red (warm subjects) or max(r,b)+2 (neutral).
Without --install it only writes previews (vlm_work/rekey_<stem>.png, on dark and light); with --install the old keyed files are kept as *_rgba.chromakey.png and the new ones replace them."""
import sys, glob, shutil, argparse, numpy as np
sys.path.insert(0, '/home/deck/workspace/csvm-hd/tools/pipeline')
from greenkey import green_key
from alpha_recipes import soft_rekey
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument('stem'); ap.add_argument('--neutral', action='store_true'); ap.add_argument('--gmax', type=float, default=None)
ap.add_argument('--no-bloom', action='store_true'); ap.add_argument('--alpha-orig', type=float, default=None, help='use the original alpha x GAIN (light flares)'); ap.add_argument('--core-only', action='store_true'); ap.add_argument('--install', action='store_true'); a = ap.parse_args()
U = glob.glob(f'/home/deck/workspace/csvm-hd/pipeline/units/{a.stem}__*')[0] + '/'
V = '/home/deck/workspace/csvm-hd/vlm_work/'
tiles, outs = [], {}
for sub in ('', 'retry/'):
    for s in (1, 2):
        try:
            rgb = np.asarray(Image.open(U + sub + f'klein_s{s}_cm.png').convert('RGB')).astype(np.float32)
        except FileNotFoundError:
            continue
        out = soft_rekey(rgb, U + 'orig.png', neutral=a.neutral, no_bloom=a.no_bloom, gmax=a.gmax, alpha_orig=a.alpha_orig, core_only=a.core_only)
        outs[(sub, s)] = out
        kk = out[..., 1].astype(int) - np.maximum(out[..., 0], out[..., 2]).astype(int); vis = out[..., 3] > 76
        print(a.stem, sub or 'first', s, 'a<8 %.2f mid %.2f opaque %.2f' % ((out[..., 3] < 8).mean(), ((out[..., 3] >= 8) & (out[..., 3] < 248)).mean(), (out[..., 3] >= 248).mean()), 'green-left %.2f%%' % (100 * ((kk > 15) & vis).sum() / max(vis.sum(), 1)))
        for c in ((25, 25, 35), (205, 210, 220)):
            im = Image.fromarray(out, 'RGBA').resize((300, 300), Image.LANCZOS); tiles.append(Image.alpha_composite(Image.new('RGBA', (300, 300), c + (255,)), im).convert('RGB'))
cols = 4; rows = (len(tiles) + cols - 1) // cols; sheet = Image.new('RGB', (cols * 305, rows * 305), (30, 30, 30))
for i, t in enumerate(tiles): sheet.paste(t, ((i % cols) * 305, (i // cols) * 305))
sheet.save(V + f'rekey_{a.stem}.png')
if a.install:
    for (sub, s), out in outs.items():
        f = U + sub + f'klein_s{s}_rgba.png'; shutil.copy(f, f[:-4] + '.chromakey.png'); Image.fromarray(out, 'RGBA').save(f)
    print('installed', len(outs), 'versions')
