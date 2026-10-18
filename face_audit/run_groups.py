import sys, json, re, collections, warnings
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw
warnings.filterwarnings("ignore")
ROOT = Path("/home/deck/workspace/csvm-hd"); sys.path.insert(0, str(ROOT / "face_audit")); sys.path.insert(0, str(ROOT / "tools"))
import transfer as T, export_winners as ew
R = json.load(open(ROOT / "face_audit/faces.json")); E = np.load(ROOT / "face_audit/embs.npy"); E = E / np.linalg.norm(E, axis=1, keepdims=True)
last = {}
for l in open(ROOT / "pipeline/reviews.jsonl"):
    if l.strip():
        r = json.loads(l); last[r["uid"]] = r
def stem(u): return re.sub(r"^(rimage_|scrap_|gfx_)", "", u.split("__")[0]).lower()
OUT = ROOT / "face_audit/out"; OUT.mkdir(exist_ok=True)
thr_group, thr_face = 0.5, 0.58
by = collections.defaultdict(dict)
for u, r in last.items():
    if u.startswith(("rimage_", "scrap_", "gfx_")) and r["decision"] == "approve" and r.get("chosen"): by[stem(u)][u] = r
def chosen_rgba(u):
    p = ew.get_chosen_file(u, last[u]["chosen"]); return np.asarray(Image.open(p).convert("RGBA")), p
def fid_unit(u, faces, W, H):
    og = [r for r in R if r["uid"] == u and r["kind"] == "orig"]
    if not og: return 0.0
    ow, oh = Image.open(ROOT / "pipeline/units" / u / "orig.png").size
    vals = []
    for f in faces:
        tc = ((f[0] + f[2] / 2) / W, (f[1] + f[3] / 2) / H)
        o = min(og, key=lambda r: np.hypot((r["box"][0] + r["box"][2] / 2) / ow - tc[0], (r["box"][1] + r["box"][3] / 2) / oh - tc[1]))
        if np.hypot((o["box"][0] + o["box"][2] / 2) / ow - tc[0], (o["box"][1] + o["box"][3] / 2) / oh - tc[1]) < 0.09:
            vals.append(float(E[o["emb_row"]] @ T.emb(np.asarray(Image.open(T.ew.get_chosen_file(u, last[u]["chosen"])).convert("RGB")), f)) if False else 0)
    return 0.0
ACCEPT = {  # (uid suffix, cx, cy): faces whose transfer I checked by eye
 "150da4a0": [(0.35,0.22)], "29b2441a": [(0.78,0.16)], "48a14f25": [(0.80,0.26)], "61619e3e": [(0.50,0.19)],
 "46ce2bcb": [(0.50,0.27)], "8efe2115": [(0.44,0.17)], "ce7f03f3": [(0.46,0.17)], "cd32a638": [(0.74,0.54)],
 "07f5097a": [(0.73,0.28),(0.45,0.16),(0.53,0.43),(0.70,0.53)], "0539e4b2": [(0.34,0.43),(0.52,0.40),(0.18,0.44)], "546bd786": [(0.47,0.38),(0.73,0.38)]}
def accepted(u, cx, cy):
    return any(abs(cx - a) < 0.025 and abs(cy - b) < 0.025 for a, b in ACCEPT.get(u[-8:], []))
def run(only=None, apply=False):
    log = []
    for s, units in sorted(by.items()):
        if only and s not in only: continue
        data = {}
        for u in units:
            rgba, p = chosen_rgba(u); faces = [f for f in T.detect(rgba[..., :3]) if f[2] >= 40]
            if faces: data[u] = (rgba, faces, p)
        if len(data) < 2: continue
        # consistency of faces across units
        allf = {u: [T.emb(d[0][..., :3], f) for f in d[1]] for u, d in data.items()}
        # fidelity of every face to the original(s) of its own unit
        fidel = {}
        for u, (rgba, faces, p) in data.items():
            og = [r for r in R if r["uid"] == u and r["kind"] == "orig"]; ow, oh = Image.open(ROOT / "pipeline/units" / u / "orig.png").size
            H, W = rgba.shape[:2]; v = []
            for f, e in zip(faces, allf[u]):
                tc = ((f[0] + f[2] / 2) / W, (f[1] + f[3] / 2) / H)
                if not og: v.append(0.0); continue
                o = min(og, key=lambda r: np.hypot((r["box"][0] + r["box"][2] / 2) / ow - tc[0], (r["box"][1] + r["box"][3] / 2) / oh - tc[1]))
                d = np.hypot((o["box"][0] + o["box"][2] / 2) / ow - tc[0], (o["box"][1] + o["box"][3] / 2) / oh - tc[1])
                v.append(float(E[o["emb_row"]] @ e) if d < 0.09 else 0.0)
            fidel[u] = v
        wmax = max(max(f[2] for f in d[1]) for d in data.values())
        cand = [u for u in data if max(f[2] for f in data[u][1]) >= 0.75 * wmax] or list(data)
        anchor = max(cand, key=lambda u: np.mean(fidel[u]))
        rgba_a, faces_a, _ = data[anchor]; Ha, Wa = rgba_a.shape[:2]
        n_tr = 0; low = []; skipped = []
        for u, (rgba, faces, p) in data.items():
            if u == anchor: continue
            H, W = rgba.shape[:2]; pairs = T.match(faces_a, faces, Wa, Ha, W, H); cur = rgba.copy(); done = []
            for fa, ft in pairs:
                cos = float(T.emb(rgba_a[..., :3], fa) @ T.emb(rgba[..., :3], ft))
                fcx, fcy = (ft[0] + ft[2] / 2) / W, (ft[1] + ft[3] / 2) / H
                if cos < thr_face and accepted(u, fcx, fcy):
                    res, info = T.transfer(rgba_a[..., :3], fa, cur, ft)
                    if res is None: skipped.append((round(cos, 2), round(info[0], 2), round(info[1], 2))); continue
                    cur = res; done.append((round(cos, 2), round(info[0], 2), round(info[1], 2)))
            if done:
                Image.fromarray(cur, "RGBA").save(OUT / f"{u}.png"); n_tr += len(done); log.append((s, u, anchor, done))
            if skipped: print("   skipped (cos, scale, landmark residual):", u[-8:], skipped)
        print(f"{s}: anchor {anchor[-8:]} (fidelity {np.mean(fidel[anchor]):.2f}); units {len(data)}; faces transferred {n_tr}", flush=True)
    json.dump(log, open(ROOT / "face_audit/transfer_log.json", "w"), indent=1)
    return log
if __name__ == "__main__":
    run(only=set(sys.argv[1:]) or None)
