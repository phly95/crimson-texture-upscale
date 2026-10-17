"""usage.py: add per-texture usage context to catalog.json (run after prep.py).

For every texture it records, over all chapters, what in the world uses it:
  nodes      top node names by polygon count (e.g. 'someroads', 'airdock1', 'pasture1')
  surface    share of its polygons that are flat-up (ground), flat-down, or vertical (walls)
  uv_repeat  median UV span of its polygons (how many times the texture repeats across a polygon)
  world_per_repeat  median world units covered by one full texture repeat (texel-density context)
  polys      polygon count
Textures with no world polygons (planes, HUD, sprites, effects) get no entry: that is itself a hint.
"""
import json, re, statistics, sys, zipfile, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXT = Path.home() / "Games/CSVM/extracted"
cat = json.load(open(ROOT / "catalog.json"))
stem_to_uid = {}
for ch, d in cat["names"].items():
    for stem, uid in d.items():
        stem_to_uid.setdefault(stem.lower(), set()).add(uid)

acc = collections.defaultdict(lambda: {"nodes": collections.Counter(), "up": 0, "down": 0, "side": 0,
                                       "uv": [], "world": [], "polys": 0})
for ch in cat["names"]:
    z = zipfile.ZipFile(EXT / ch / "gamez.zip")
    mats = json.loads(z.read("materials.json"))
    texs = [t["name"].rsplit(".", 1)[0].lower() for t in json.loads(z.read("textures.json"))]
    models = json.loads(z.read("models.json"))
    nodes = json.loads(z.read("nodes.json"))
    node_names = collections.defaultdict(list)
    for n in nodes:
        if n.get("model_index") is not None and n["model_index"] >= 0:
            node_names[n["model_index"]].append(re.sub(r"\d+$", "", n["name"]) or n["name"])
    mat_tex = {}
    for i, m in enumerate(mats):
        t = m.get("Textured")
        if t and t["texture_index"] < len(texs):
            mat_tex[i] = texs[t["texture_index"]]
    for mi, m in enumerate(models):
        verts = m.get("vertices") or []
        if not verts:
            continue
        V = np.array([[v["x"], v["y"], v["z"]] for v in verts], dtype=np.float64)
        names = node_names.get(mi) or ["(unnamed)"]
        for p in m["polygons"]:
            vi = p["vertex_indices"]
            if len(vi) < 3:
                continue
            P = V[vi]
            nrm = np.cross(P[1] - P[0], P[2] - P[0])
            area = float(np.linalg.norm(nrm)) / 2.0
            if area <= 0:
                continue
            ny = float(nrm[1] / (np.linalg.norm(nrm) + 1e-12))
            for mt in p.get("materials", []):
                tex = mat_tex.get(mt["material_index"])
                uvs = mt.get("uv_coords") or []
                if tex is None or len(uvs) < 3:
                    continue
                a = acc[tex]
                a["polys"] += 1
                for nm in names[:2]:
                    a["nodes"][nm] += 1
                a["up" if ny > 0.7 else "down" if ny < -0.7 else "side"] += 1
                u = [c["u"] for c in uvs]; v = [c["v"] for c in uvs]
                a["uv"].append((max(u) - min(u), max(v) - min(v)))
                U = np.array([[c["u"], c["v"]] for c in uvs[:3]])
                d1, d2 = U[1] - U[0], U[2] - U[0]
                uv_area = abs(d1[0] * d2[1] - d1[1] * d2[0]) / 2.0
                if uv_area > 1e-9:
                    a["world"].append((area / uv_area) ** 0.5)
    print(ch, "done", flush=True)

n_added = 0
for stem, a in acc.items():
    if a["polys"] == 0:
        continue
    tot = a["up"] + a["down"] + a["side"]
    rec = {
        "polys": a["polys"],
        "nodes": [f"{k}({v})" for k, v in a["nodes"].most_common(6)],
        "surface": {"ground_up": round(a["up"] / tot, 2), "ceiling_down": round(a["down"] / tot, 2),
                    "wall_side": round(a["side"] / tot, 2)},
        "uv_repeat_median": [round(statistics.median(x[0] for x in a["uv"]), 2),
                             round(statistics.median(x[1] for x in a["uv"]), 2)],
        "world_units_per_repeat": round(statistics.median(a["world"]), 1) if a["world"] else None,
    }
    for uid in stem_to_uid.get(stem, ()):
        cat["units"][uid]["usage"] = rec
        n_added += 1
json.dump(cat, open(ROOT / "catalog.json", "w"), indent=1)
print("units with usage:", n_added, "of", len(cat["units"]))
