"""Independent decode of a ZBD texture package (RGB565 / palette, no compression) and compare to extracted PNGs."""
import struct, sys, zipfile, io
import numpy as np
from PIL import Image
zbd, zipp = sys.argv[1], sys.argv[2]
d = open(zbd, 'rb').read()
_, has, gp, n, _, _ = struct.unpack('<6I', d[:24])
gp = struct.unpack('<6i', d[:24])[2]
ents = []
for i in range(n):
    o = 24 + 40*i
    name = d[o:o+32].split(b'\0')[0].decode()
    off, pal = struct.unpack('<Ii', d[o+32:o+40])
    ents.append((name, off, pal))
z = zipfile.ZipFile(zipp)
stats = dict(n=0, exact=0, diff=0, pal=0, nonfull=0)
maxdiff = []
levels = [set(), set(), set()]
for name, off, pal in ents:
    flags, w, h, _, pc, _ = struct.unpack('<IHHIHH', d[off:off+16])
    p = off + 16
    if pc != 0 or pal != -1:
        stats['pal'] += 1
        continue
    px = np.frombuffer(d, '<u2', w*h, p); p += w*h*2
    a = None
    if flags & 0x08:
        a = np.frombuffer(d, np.uint8, w*h, p); p += w*h
    elif flags & 0x02:
        a = np.where(px == 0, 0, 255).astype(np.uint8)
    r5, g6, b5 = (px >> 11) & 31, (px >> 5) & 63, px & 31
    # mech3ax lerp: x*255/31 rounded
    rgb = np.stack([np.round(r5*255/31), np.round(g6*255/63), np.round(b5*255/31)], -1).astype(np.uint8).reshape(h, w, 3)
    ref = np.asarray(Image.open(io.BytesIO(z.read(name + '.png'))).convert('RGBA'))
    stats['n'] += 1
    dm = np.abs(ref[..., :3].astype(int) - rgb.astype(int)).max()
    maxdiff.append(dm)
    if a is not None:
        da = np.abs(ref[..., 3].astype(int) - a.reshape(h, w).astype(int)).max()
        if da: stats['diff'] += 1
    for c in range(3):
        levels[c].update(np.unique(ref[..., c]).tolist())
print(stats, 'max rgb diff vs PNG over all:', max(maxdiff), 'mean', np.mean(maxdiff))
print('distinct levels in extracted PNG R/G/B:', [len(s) for s in levels])
