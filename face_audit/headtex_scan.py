"""headtex_scan.py [OUT.json]: scan every model of every pack for textures that are UV-mapped onto 3D geometry (not tiled, not flat) and flag wrap defects:
collapsed UV triangles (density < 0.05 x the model's median: a thin strip of texels smeared over an area), extreme texel-density spread (stretch), UV islands that overlap each other (shared/mirrored texels),
and how much of the sheet the UV islands use (atlas-like). One row per texture stem, aggregated over models/chapters."""
import json, sys, zipfile, collections
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]; EX = Path.home() / "Games/CSVM/extracted"
KNOWN = set("cphead cphead02 womanhead womanface fbhead1 cpsuit128 cpsuit128_01 cpsuit128_02 womanchest womanarm woman1 fbsuit128_01 fbsuit128_02 pass_bswan pass_fas pass_jack pass_john pass_pick pass_sparks pass_waldo ilsafas uboothe pass_coat1 pass_coat2 sweater".split())
def load(pack):
    z = zipfile.ZipFile(EX / pack); tn = [t["name"].rsplit(".", 1)[0].lower() for t in json.loads(z.read("textures.json"))]
    return json.loads(z.read("models.json")), json.loads(z.read("materials.json")), tn
def model_tris(m, mats, tn):
    V = np.array([[v["x"], v["y"], v["z"]] for v in m["vertices"]], np.float32); T = []
    for p in m["polygons"]:
        mt = mats[p["materials"][0]["material_index"]].get("Textured")
        if not mt: continue
        name = tn[mt["texture_index"]]; vi = p["vertex_indices"]; uv = [(u["u"], u["v"]) for u in p["materials"][0]["uv_coords"]]
        if p["flags"]["tri_strip"]:
            for i in range(len(vi) - 2):
                a = (vi[i], vi[i + 1], vi[i + 2]) if i % 2 == 0 else (vi[i + 1], vi[i], vi[i + 2]); u = (uv[i], uv[i + 1], uv[i + 2]) if i % 2 == 0 else (uv[i + 1], uv[i], uv[i + 2])
                if len(set(a)) == 3: T.append((a, u, name))
        else:
            for i in range(1, len(vi) - 1): T.append(((vi[0], vi[i], vi[i + 1]), (uv[0], uv[i], uv[i + 1]), name))
    return V, T
def analyse(V, T):
    P = [V[list(vi)] for vi, uv, nm in T]; a3 = np.array([np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) / 2 for p in P])
    a2 = np.array([abs((u[1][0] - u[0][0]) * (u[2][1] - u[0][1]) - (u[1][1] - u[0][1]) * (u[2][0] - u[0][0])) / 2 for vi, u, nm in T])
    ok = a3 > 1e-9; dens = np.where(ok, a2 / np.maximum(a3, 1e-12), np.nan); med = np.nanmedian(dens[ok]) if ok.any() else 1.0
    return dens / med, a3
_CAT = {v["stem"]: (v["w"], v["h"]) for v in json.load(open(ROOT / "catalog.json"))["units"].values() if not v.get("is_mip")}
def slivers(T, js, a3):
    """true collapsed UVs: the UV triangle is < 0.75 texel wide yet longer than 4 texels, while its 3D area is at least 10 % of the texture's median 3D area in the model"""
    w, h = _CAT.get(T[js[0]][2], (128, 128)); med = np.median(a3[js][a3[js] > 1e-9]) if (a3[js] > 1e-9).any() else 0; n = 0; vis = 0.0
    for j in js:
        u = np.array(T[j][1]) * [w, h]; e = [np.linalg.norm(u[i] - u[(i + 1) % 3]) for i in range(3)]; area = abs((u[1][0] - u[0][0]) * (u[2][1] - u[0][1]) - (u[1][1] - u[0][1]) * (u[2][0] - u[0][0])) / 2
        if max(e) >= 4 and 2 * area / max(e) < 0.75 and a3[j] >= 0.1 * med > 0: n += 1; vis += a3[j]
    return n, vis
def run(out):
    rows = collections.defaultdict(lambda: dict(vis=0.0, models=set(), packs=set(), tris=0, broken=0, spread=[], overlap=[], cover=[], flat=[], inrange=[], dims=[], tex_tris=0))
    for pack in ["planes.zip"] + [f"{c}/gamez.zip" for c in sorted(p.name for p in EX.iterdir() if (p / "gamez.zip").exists())]:
        models, mats, tn = load(pack); seen = set()
        for mi, m in enumerate(models):
            V, T = model_tris(m, mats, tn)
            if not T: continue
            key = hash((V.tobytes(), tuple(t[2] for t in T), tuple(round(q, 4) for t in T[:40] for uv in t[1] for q in uv)))
            if key in seen: continue
            seen.add(key); rel, a3 = analyse(V, T); ext = V.max(0) - V.min(0); flat = float(ext.min() / max(ext.max(), 1e-9))
            by = collections.defaultdict(list)
            for j, (vi, uv, nm) in enumerate(T): by[nm].append(j)
            for nm, js in by.items():
                r = rows[nm]; r["models"].add(f"{pack}:{mi}"); r["packs"].add(pack.split("/")[0]); r["tris"] += len(js)
                uvs = np.array([T[j][1] for j in js]); r["inrange"].append(bool(uvs.min() >= -0.01 and uvs.max() <= 1.01))
                d = rel[js]; d = d[np.isfinite(d)]; sn, sv = slivers(T, js, a3); r["broken"] += sn; r["vis"] = r.get("vis", 0.0) + sv
                if len(d) >= 4: dd = d[d >= 0.05]; r["spread"].append(float(np.percentile(dd, 95) / max(np.percentile(dd, 5), 1e-9)) if len(dd) >= 4 else 0.0)
                g = np.zeros((128, 128), np.uint16)
                for j in js:
                    q = (np.array(T[j][1]) * 127 % 128 if False else np.clip(np.array(T[j][1]), 0, 1) * 127).astype(np.int32); m_ = np.zeros((128, 128), np.uint8); cv2.fillPoly(m_, [q[:, None, :]], 1); g += m_
                cov = float((g > 0).mean()); r["cover"].append(cov); r["overlap"].append(float((g > 1).sum() / max((g > 0).sum(), 1))); r["flat"].append(flat)
    res = []
    for nm, r in rows.items():
        res.append(dict(stem=nm, packs=sorted(r["packs"]), models=len(r["models"]), tris=r["tris"], broken=r["broken"], vis=round(float(r["vis"]), 3), spread=round(float(np.max(r["spread"])) if r["spread"] else 0, 1),
                        overlap=round(float(np.max(r["overlap"])), 2), cover=round(float(np.min(r["cover"])), 2), flat=round(float(np.median(r["flat"])), 3), inrange=float(np.mean(r["inrange"])), known=nm in KNOWN))
    json.dump(res, open(out, "w")); print(len(res), "textures ->", out)
if __name__ == "__main__": run(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "face_audit/headtex_scan.json"))
