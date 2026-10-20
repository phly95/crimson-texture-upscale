"""headtex_prop.py TARGET: identity propagation. 1) bake the (portrait) front view alone over the old full bake; 2) render the non-front views of the mesh from that texture and let klein restore them (they keep the new face);
3) bake all views -> headtex/<stem>_mesh.png.  Needs r_<tg>_front.png (the composed front) and the old <stem>_mesh.png (or fallback) as base."""
import sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_run as R, headtex_bake as B, headtex_fin as F
T, D = R.T, R.D
def tex_dict(C, fills):
    return {name: fills[name] for name, *_ in C["bake"]}
def run(tg):
    C = R.TARGETS[tg]; V = R.VIEWS; zb = {k: np.load(D / f"v_{tg}_{k}_z.npy") for k in V}
    front = (V["front"][0], V["front"][1], np.array(Image.open(D / f"r_{tg}_front.png").convert("RGB")), zb["front"])
    fills = {}
    for name, stem, (tw, th), fbk in C["bake"]:                                                                  # step 1: front-only bake over the old texture
        old = np.array(Image.open(D / f"{stem}_mesh.png").convert("RGB").resize((tw, th), Image.LANCZOS)).astype(np.float64)
        b, w = B.bake([front], model=C["model"], pack=C["pack"], only=name, TS=tw, TSH=th)
        a = np.clip(w / 0.25, 0, 1)[..., None]; fills[name] = np.clip(b * a + old * (1 - a) + .5, 0, 255).astype(np.uint8)
    views = [front]
    for k, (y, p) in V.items():
        if k == "front": continue
        texd = {n: fills[n] for n in fills}; tx = {t[2]: texd.get(t[2], next(iter(texd.values()))) for t in B.HR.tris(C["model"], C["pack"])[1] if t[2]}
        # materials not baked (e.g. womanarm) keep the original texture
        for n, im in C["tex"].items():
            if n not in texd: tx[n] = im
        im, z = B.render_view(tx, y, p, model=C["model"], pack=C["pack"]); src = D / f"v2_{tg}_{k}.png"; Image.fromarray(im).save(src); out = D / f"g2_{tg}_{k}.png"; fin = D / f"r2_{tg}_{k}.png"
        prompt = "This is " + C["who"] + R.COMMON + R.NOFACE.get(k, "")
        if not fin.exists():
            T.locked_gen_wh(str(src), str(out), prompt, 1024, 1024, 7)
            ref = im; gen = np.array(Image.open(out).convert("RGB").resize((1024, 1024), Image.LANCZOS)); w_, fl = F.align(gen, ref, smooth=14.0, clip=30.0)
            Image.fromarray((np.clip(T.colour_lock(w_ / 255., ref / 255., 10.0, 30.0), 0, 1) * 255 + .5).astype(np.uint8)).save(fin)
        print(tg, k, "restored", flush=True); views.append((y, p, np.array(Image.open(fin).convert("RGB")), zb[k]))
    for name, stem, (tw, th), fbk in C["bake"]:                                                                  # step 3: bake everything
        b, w = B.bake(views, model=C["model"], pack=C["pack"], only=name, TS=tw, TSH=th)
        a = np.clip(w / 0.25, 0, 1)[..., None]; o = b * a + fills[name].astype(np.float64) * (1 - a)
        Image.fromarray(np.clip(o + .5, 0, 255).astype(np.uint8)).save(D / f"{stem}_mesh2.png"); print(stem, "covered %.2f" % (w > 0.05).mean())
if __name__ == "__main__": run(sys.argv[1])
