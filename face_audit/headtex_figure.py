"""headtex_figure.py: whole-figure meshes (node tree of a pilot figure: head + torso + limbs, translations only) for headtex_render, plus render sheets.
fig(pack, root_node) -> model key usable as HR.render(..., model=key, pack=pack); texdict(stems) -> dict of original textures (missing ones grey);
sheet(rows, model, pack, out, S) -> labelled rows (label, tex_dict) at 6 yaws (front 180, 3/4 135/225, profiles 90/270, back 0)"""
import json, sys, zipfile
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headtex_render as HR
ROOT = HR.ROOT; cat = json.load(open(ROOT / "catalog.json"))["units"]
FIGS = {"pilot": ("planes.zip", 155), "copilot": ("planes.zip", 1035), "fbman": ("C1C/gamez.zip", 2282), "woman": ("C1/gamez.zip", 2673)}
def fig(name, fixuv=False):
    """fixuv: collapsed-UV triangles of every node model borrow their mirror twin's UVs (per model, so the mirror search cannot pair vertices of different parts)"""
    pack, root = FIGS[name]; key = "fig:" + name + (":fix" if fixuv else "")
    if (pack, key) in HR._M: return key, pack
    z = zipfile.ZipFile(Path.home() / "Games/CSVM/extracted" / pack); N = json.loads(z.read("nodes.json")); Vs, Ts, off = [], [], 0
    def walk(i, t):
        nonlocal off
        n = N[i]; tr = (n.get("data") or {}).get("Object3d", {}).get("transform", {})
        if isinstance(tr, dict) and "RotateTranslateScale" in tr: d = tr["RotateTranslateScale"]["translate"]; t = t + np.array([d["x"], d["y"], d["z"]], np.float32)
        if n["model_index"] >= 0:
            V, T = HR.tris(n["model_index"], pack, fixuv); Vs.append(V + t); Ts.extend([(tuple(x + off for x in vi), uv, nm) for vi, uv, nm in T]); off += len(V)
        for c in n["child_indices"]: walk(c, t)
    walk(root, np.zeros(3, np.float32)); HR._M[(pack, key)] = (np.concatenate(Vs), Ts); return key, pack
def orig(stem):
    u = next((u for u, v in cat.items() if v["stem"] == stem and not v.get("is_mip")), None)
    return None if u is None else np.array(Image.open(ROOT / f"pipeline/units/{u}/orig.png").convert("RGB"))
class TexDict(dict):
    def __missing__(self, k):
        t = orig(k) if k else None; self[k] = t if t is not None else np.full((8, 8, 3), 128, np.uint8); return self[k]
def texdict(over=None): d = TexDict(); d.update(over or {}); return d
YAWS = (180, 135, 90, 0, 270, 225)
def sheet(rows, model, pack, out, S=420, yaws=YAWS, pitch=0, fixuv=False):
    if fixuv and model.startswith("fig:") and not model.endswith(":fix"): model = fig(model[4:], True)[0]; fixuv = False
    ims = []
    for lab, tex in rows:
        r = np.hstack([HR.render(tex, y, pitch, S=S, flipv=False, model=model, pack=pack, engine=True, fixuv=fixuv) for y in yaws]); cv2.putText(r, lab, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA); ims.append(r)
    Image.fromarray(np.vstack(ims)).save(out); return out
