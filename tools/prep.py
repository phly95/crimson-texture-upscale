"""Extract each chapter's top-tier texture set, dedupe by content, and build catalog.json + tasks.json.

Reads  ~/Games/CSVM/extracted/<CH>/{rtextureN.zip (highest N), gamez.zip}
Writes <ws>/originals/<CH>/*.png (+ manifest.json), <ws>/catalog.json, <ws>/tasks/tasks.json

A "unit" is one unique texture content (sha1 of the PNG bytes). The pack maps every
(chapter, stem) back to its unit, so a texture shared by six chapters is upgraded once.
Mip siblings follow the engine's own rule (TextureArchive: `X_1`/`X_2` is a mip iff `X` exists).
"""
import glob, hashlib, json, os, re, struct, sys, zipfile, collections

EXT = os.path.expanduser("~/Games/CSVM/extracted")
ws = sys.argv[1]
MAX_GROUP = 8


def safe(name):
    b = os.path.basename(name)
    assert b and b not in (".", "..") and "/" not in b and "\\" not in b
    return b


def top_tier(ch_dir):
    best = max((int(re.sub(r"\D", "", os.path.basename(z)[8:-4]) or 0), z)
               for z in glob.glob(ch_dir + "/rtexture*.zip"))
    return best


chapters = [c for c in sorted(os.listdir(EXT)) if re.fullmatch(r"C\d[A-Z]?", c) and os.path.isdir(f"{EXT}/{c}")]
units = {}                       # sha1 -> unit dict
names_of = collections.defaultdict(dict)   # chapter -> stem -> sha1
cycle_edges = []                 # (stem_a, stem_b) frames that belong together
tier_of = {}

for ch in chapters:
    n, z = top_tier(f"{EXT}/{ch}")
    tier_of[ch] = f"rtexture{n}"
    out = f"{ws}/originals/{ch}"
    os.makedirs(out, exist_ok=True)
    zf = zipfile.ZipFile(z)
    manifest = {}
    stems = {}
    for nm in zf.namelist():
        b = safe(nm)
        if nm.endswith("/"):
            continue
        data = zf.read(nm)
        open(f"{out}/{b}", "wb").write(data)
        if b.lower() == "manifest.json":
            manifest = json.loads(data)
            continue
        if not b.lower().endswith(".png"):
            continue
        stem = b[:-4]
        h = hashlib.sha1(data).hexdigest()
        w, hh = struct.unpack(">II", data[16:24])
        stems[stem] = (h, w, hh, data[25])
    info = {t["name"].rsplit(".", 1)[0].lower(): t for t in manifest.get("texture_infos", [])}
    for stem, (h, w, hh, ctype) in stems.items():
        names_of[ch][stem] = h
        u = units.setdefault(h, {"sha1": h, "w": w, "h": hh, "png_color_type": ctype, "stems": {}, "chapters": []})
        u["stems"].setdefault(stem, []).append(ch)
        if ch not in u["chapters"]:
            u["chapters"].append(ch)
        t = info.get(stem.lower())
        if t and "alpha" not in u:
            u["alpha"] = t.get("alpha")
            u["stretch"] = t.get("stretch")
    # mip siblings, engine rule
    for stem in stems:
        m = re.search(r"_([12])$", stem)
        if m and stem[:-2] in stems:
            units[stems[stem][0]].setdefault("mip_of", set()).add(stem[:-2])
    # flipbook groups from the world's material table
    gz = zipfile.ZipFile(f"{EXT}/{ch}/gamez.zip")
    mats = json.loads(gz.read("materials.json"))
    texs = [t["name"].rsplit(".", 1)[0] for t in json.loads(gz.read("textures.json"))]
    for mt in mats:
        c = mt.get("Textured", {}).get("cycle")
        if c:
            idx = [texs[i] for i in c["texture_indices"] if i < len(texs)]
            for a, b in zip(idx, idx[1:]):
                cycle_edges.append((a.lower(), b.lower()))
    print(f"{ch}: {tier_of[ch]} {len(stems)} pngs", flush=True)

# case-insensitive stem -> unit sha, for cycle lookups
lower_to_sha = {}
for ch, d in names_of.items():
    for stem, h in d.items():
        lower_to_sha.setdefault(stem.lower(), set()).add(h)

# union-find over units for flipbooks
parent = {h: h for h in units}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for a, b in cycle_edges:
    for ha in lower_to_sha.get(a, ()):
        for hb in lower_to_sha.get(b, ()):
            parent[find(ha)] = find(hb)

PLACEHOLDER = re.compile(r"_(nose|tail|wing)logo$", re.I)
for h, u in units.items():
    stems = sorted(u["stems"])
    u["stem"] = stems[0]
    u["uid"] = f"{stems[0]}__{h[:8]}"
    u["mip_of"] = sorted(u.get("mip_of", []))
    u["is_mip"] = bool(u["mip_of"])
    u["is_placeholder"] = bool(PLACEHOLDER.search(u["stem"]))
    u["has_alpha"] = u["png_color_type"] == 6
    u["target"] = [min(u["w"] * 8, 2048), min(u["h"] * 8, 2048)]
    u["flip"] = None

# flipbook membership
members = collections.defaultdict(list)
for h in units:
    members[find(h)].append(h)
flip_groups = [sorted(v) for v in members.values() if len(v) > 1]
for g in flip_groups:
    for h in g:
        units[h]["flip"] = sorted(units[x]["uid"] for x in g)[0]

work = [u for u in units.values() if not u["is_mip"] and not u["is_placeholder"]]


def family(u):
    s = re.sub(r"^\d+", "", u["stem"].lower())
    m = re.match(r"[a-z]+", s)
    return (m.group(0) if m else "misc")[:6]


# tasks: flipbooks are indivisible; others chunk by family
tasks = []
used = set()
for g in flip_groups:
    gu = [units[h] for h in g if not units[h]["is_mip"] and not units[h]["is_placeholder"]]
    if not gu:
        continue
    for u in gu:
        used.add(u["sha1"])
    tasks.append({"kind": "flipbook", "family": family(gu[0]), "units": sorted(u["uid"] for u in gu)})
fam = collections.defaultdict(list)
for u in sorted(work, key=lambda u: u["stem"].lower()):
    if u["sha1"] not in used:
        fam[family(u)].append(u["uid"])
for f, ids in sorted(fam.items()):
    for i in range(0, len(ids), MAX_GROUP):
        chunk = ids[i:i + MAX_GROUP]
        tasks.append({"kind": "family", "family": f, "units": chunk})
# merge tiny tails: families with <=2 units get pooled into "misc" chunks
small = [t for t in tasks if t["kind"] == "family" and len(t["units"]) <= 2]
tasks = [t for t in tasks if t not in small]
pool = [uid for t in small for uid in t["units"]]
for i in range(0, len(pool), MAX_GROUP):
    tasks.append({"kind": "misc", "family": "misc", "units": pool[i:i + MAX_GROUP]})
for i, t in enumerate(tasks):
    t["id"] = f"t{i:04d}"

cat = {
    "tiers": tier_of,
    "units": {u["uid"]: {k: v for k, v in u.items() if k != "sha1"} | {"sha1": u["sha1"]} for u in units.values()},
    "names": {ch: {s: units[h]["uid"] for s, h in d.items()} for ch, d in names_of.items()},
}
json.dump(cat, open(f"{ws}/catalog.json", "w"), indent=1)
os.makedirs(f"{ws}/tasks", exist_ok=True)
json.dump(tasks, open(f"{ws}/tasks/tasks.json", "w"), indent=1)

print("units:", len(units), "| mips:", sum(u["is_mip"] for u in units.values()),
      "| placeholders:", sum(u["is_placeholder"] for u in units.values()), "| work units:", len(work))
print("flipbook groups:", len(flip_groups), "sizes:", sorted(len(g) for g in flip_groups)[-8:])
print("tasks:", len(tasks), " sizes:", collections.Counter(len(t["units"]) for t in tasks).most_common())
print("alpha classes:", collections.Counter(u.get("alpha") for u in work))
