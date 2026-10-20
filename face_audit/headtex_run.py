"""headtex_run.py TARGET [steps]: mesh-based redraw of a head texture. TARGET in cphead02 woman fbhead1 (cphead was done by headtex_views/headtex_bakeall).
Renders the head mesh with the ORIGINAL texture from 7 views, klein-redraws each view (aligned + colour-locked to the render), bakes the views back into the UV layout (headtex_bake),
fills gaps from the layout-locked redraw -> headtex/<stem>_mesh.png"""
import sys, json
from pathlib import Path
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_bake as B, headtex_fin as F
T = F.T; ROOT = F.H.ROOT; D = ROOT / "face_audit/headtex"
cat = json.load(open(ROOT / "catalog.json"))["units"]
def orig(stem):
    u = next(u for u, v in cat.items() if v["stem"] == stem and not v.get("is_mip")); return np.array(Image.open(ROOT / f"pipeline/units/{u}/orig.png").convert("RGB"))
VIEWS = {"front": (180, 0), "right": (90, 0), "left": (270, 0), "back": (0, 0), "top": (0, 70), "q1": (135, 15), "q2": (225, 15)}
NOFACE = {"back": " This is the BACK of the head seen from behind: only the back of the head, hair or cap, and the nape of the neck are visible; there is no face, no eyes and no mouth anywhere.",
          "top": " This is the TOP of the head seen from directly above: only the top of the hair or cap is visible; there is no face, no eyes and no mouth anywhere."}
COMMON = (" Restore this render as a crisp, sharp, high-resolution, photorealistic version of itself. Keep every shape, colour, position and outline exactly the same; do not re-frame, re-scale, rotate or move anything, "
          "keep the plain grey background, add only fine realistic surface detail and remove blur and pixelation. Flat, even lighting with no cast shadows.")
TARGETS = {
    "cphead": dict(pack="planes.zip", model=70, tex={"cphead": orig("cphead")}, bake=[("cphead", "cphead", (1024, 1024), "cphead_s2_fin.png")],
                   who="the head and neck of a 1930s aviator in a stitched dark brown leather flying helmet with chinstrap and round goggles, realistic skin and leather."),
    "cphead02": dict(pack="planes.zip", model=70, tex={"cphead": orig("cphead02")}, bake=[("cphead", "cphead02", (1024, 1024), "cphead02_s2_fin.png")],
                     who="the head and neck of a 1930s aviator in a stitched dark brown leather flying helmet with big ear flaps, strap and round goggles, realistic skin and leather."),
    "woman": dict(pack="C1/gamez.zip", model=1175, tex={"womanhead": orig("womanhead"), "womanface": orig("womanface"), "womanarm": orig("womanarm")},
                  bake=[("womanhead", "womanhead", (1024, 1024), "womanhead_s1_fin.png"), ("womanface", "womanface", (1024, 4096), "womanface_s2_fin.png")],
                  who="the head and neck of a smiling young woman with dark brown skin, wearing a snug plain dark leather skull cap, realistic smooth hairless skin: no beard, no stubble, no moustache, no hair on the face or chin, bright natural white teeth."),
    "fbhead1": dict(pack="C1C/gamez.zip", model=1124, tex={"fbhead1": orig("fbhead1")}, bake=[("fbhead1", "fbhead1", (1024, 1024), "fbhead1_s1_fin.png")],
                    who="the head and neck of a smiling elderly man with round glasses and a white woolly cap, realistic skin, wrinkles and wool."),
}
def run(tg, steps="rbc"):
    C = TARGETS[tg]; views = []
    for k, (y, p) in VIEWS.items():
        src, out, fin = D / f"v_{tg}_{k}.png", D / f"g_{tg}_{k}.png", D / f"r_{tg}_{k}.png"
        if "r" in steps:
            im, zb = B.render_view(C["tex"], y, p, model=C["model"], pack=C["pack"]); Image.fromarray(im).save(src); np.save(D / f"v_{tg}_{k}_z.npy", zb)
        if "c" in steps and not fin.exists():
            prompt = "This is " + C["who"] + COMMON + NOFACE.get(k, "")
            T.locked_gen_wh(str(src), str(out), prompt, 1024, 1024, 7)
            ref = np.array(Image.open(src).convert("RGB")); gen = np.array(Image.open(out).convert("RGB").resize((1024, 1024), Image.LANCZOS))
            w, fl = F.align(gen, ref, smooth=14.0, clip=30.0); fn = np.clip(T.colour_lock(w / 255., ref / 255., 10.0, 30.0), 0, 1)
            Image.fromarray((fn * 255 + .5).astype(np.uint8)).save(fin); print(tg, k, "lowcorr %.3f" % F.lowcorr((fn * 255).astype(np.uint8), ref), flush=True)
        if "b" in steps: views.append((y, p, np.array(Image.open(fin).convert("RGB")), np.load(D / f"v_{tg}_{k}_z.npy")))
    if "b" in steps:
        for name, stem, (tw, th), fbk in C["bake"]:
            b, w = B.bake(views, model=C["model"], pack=C["pack"], only=name, TS=tw, TSH=th)
            fb = np.array(Image.open(D / fbk).convert("RGB").resize((b.shape[1], b.shape[0]), Image.LANCZOS)).astype(np.float64)
            a = np.clip(w / 0.25, 0, 1)[..., None]; o = b * a + fb * (1 - a)
            Image.fromarray(np.clip(o + .5, 0, 255).astype(np.uint8)).save(D / f"{stem}_mesh.png"); print(stem, "covered %.2f" % (w > 0.05).mean())
if __name__ == "__main__": run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "rbc")
