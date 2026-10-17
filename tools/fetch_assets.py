"""Fetch CC0 albedo/color maps from ambientCG and Poly Haven into assets/, with a provenance index.

Resumable: existing, valid files are skipped. Every download is decoded with Pillow before it is
kept, and each kept file gets one line in assets/index.jsonl (source, id, licence, url, tags, size,
mean colour). Both libraries publish under CC0-1.0.

usage: fetch_assets.py <workspace> [--sources ambientcg,polyhaven] [--workers 6]
"""
import argparse, json, os, sys, time, threading, urllib.request, urllib.error, io
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import numpy as np

UA = {"User-Agent": "csvm-hd-texture-research/0.1 (personal hobby project; CC0 assets only)"}
Image.MAX_IMAGE_PIXELS = 100_000_000

AMBIENT_CATS = {
    # always take everything in these (small categories, high relevance)
    "all": {"Ground", "Gravel", "Grass", "Rock", "Rocks", "Asphalt", "Road", "Bricks", "Painted Bricks",
            "Concrete", "Plaster", "Painted Plaster", "Metal", "Metal Plates", "Painted Metal",
            "Diamond Plate", "Corrugated Steel", "Sheet Metal", "Rust", "Wood", "Planks", "Painted Wood",
            "Wood Siding", "Bark", "Roofing Tiles", "Facade", "Snow", "Moss", "Granite", "Fence", "Net",
            "Paper", "Cardboard", "Pipe", "Metal Walkway", "Grate", "Chainmail", "Scratches",
            "Surface Imperfections", "Wood Floor", "Sign", "Thatched Roof", "Clay", "Rails", "Pathway",
            # second pass, from what the pilot agents asked for / what a 1930s diesel-punk world needs
            "Rubber", "Paint", "Smear", "Fingerprints", "Chip", "Ice", "Rope", "Onyx", "Travertine",
            "Glazed Terracotta", "Plastic", "Foil", "Porcelain", "Styrofoam", "Bark"},
    # big categories: top-N by popularity
    "top": {"Tiles": 40, "Paving Stones": 40, "Fabric": 30, "Leather": 15, "Marble": 10, "Carpet": 8,
            "Wallpaper": 6, "Terrazzo": 6, "Wicker": 6},
}
POLY_SKIP_CATS = {"denim", "cotton", "wool", "fleece", "stretchy"}


def get(url, tries=4, timeout=60):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def validate_and_stats(data):
    im = Image.open(io.BytesIO(data))
    im.load()
    rgb = im.convert("RGB")
    small = rgb.resize((32, 32))
    mean = np.asarray(small, dtype=np.float32).reshape(-1, 3).mean(0)
    return im.size, [round(float(x), 1) for x in mean]


def select_ambient(catalog):
    L = json.load(open(catalog))["foundAssets"]
    out = []
    top = {}
    for m in L:
        cat = m["displayCategory"]
        if cat in AMBIENT_CATS["all"]:
            out.append(m)
        elif cat in AMBIENT_CATS["top"]:
            top.setdefault(cat, []).append(m)
    for cat, n in AMBIENT_CATS["top"].items():
        out += sorted(top.get(cat, []), key=lambda m: -m["popularityScore"])[:n]
    jobs = []
    for m in out:
        aid = m["assetId"]
        jobs.append({
            "source": "ambientcg", "id": aid,
            "url": f"https://f003.backblazeb2.com/file/ambientCG-Web/media/surface-preview/{aid}/{aid}_SQ_Color.jpg",
            "page": m.get("shortLink"), "category": m["displayCategory"], "tags": m.get("tags", []),
            "name": m["displayName"], "license": "CC0-1.0", "ext": "jpg",
        })
    return jobs


def select_poly(catalog, res="2k"):
    P = json.load(open(catalog))
    jobs = []
    for pid, v in P.items():
        cats = set(v.get("categories", []))
        if cats & POLY_SKIP_CATS:
            continue
        jobs.append({
            "source": "polyhaven", "id": pid,
            "url": f"https://dl.polyhaven.org/file/ph-assets/Textures/jpg/{res}/{pid}/{pid}_diff_{res}.jpg",
            "page": f"https://polyhaven.com/a/{pid}", "category": v.get("category") or "/".join(sorted(cats)),
            "tags": v.get("tags", []) + sorted(cats), "name": v["name"], "license": "CC0-1.0", "ext": "jpg",
        })
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ws")
    ap.add_argument("--sources", default="ambientcg,polyhaven")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="cap jobs per source (testing)")
    a = ap.parse_args()
    ws = a.ws
    cat = os.path.join(ws, "assets", "_catalog")
    jobs = []
    if "ambientcg" in a.sources:
        jobs += select_ambient(os.path.join(cat, "ambientcg.json"))[: a.limit or None]
    if "polyhaven" in a.sources:
        jobs += select_poly(os.path.join(cat, "polyhaven.json"))[: a.limit or None]
    index_path = os.path.join(ws, "assets", "index.jsonl")
    done = set()
    if os.path.exists(index_path):
        for line in open(index_path):
            r = json.loads(line)
            done.add((r["source"], r["id"]))
    todo = [j for j in jobs if (j["source"], j["id"]) not in done]
    print(f"jobs={len(jobs)} already={len(jobs)-len(todo)} todo={len(todo)}", flush=True)
    lock = threading.Lock()
    fails = []
    t0 = time.time()
    nbytes = 0

    def work(j):
        d = os.path.join(ws, "assets", j["source"])
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{j['id']}.{j['ext']}")
        data = get(j["url"])
        size, mean = validate_and_stats(data)
        with open(path, "wb") as f:
            f.write(data)
        rec = {k: j[k] for k in ("source", "id", "name", "category", "tags", "license", "url", "page")}
        rec.update({"path": os.path.relpath(path, ws), "w": size[0], "h": size[1], "mean_rgb": mean,
                    "bytes": len(data)})
        return rec

    with ThreadPoolExecutor(a.workers) as ex, open(index_path, "a") as idx:
        futs = {ex.submit(work, j): j for j in todo}
        for n, fu in enumerate(as_completed(futs), 1):
            j = futs[fu]
            try:
                rec = fu.result()
                with lock:
                    idx.write(json.dumps(rec) + "\n"); idx.flush()
                    nbytes += rec["bytes"]
            except Exception as e:  # noqa
                fails.append((j["source"], j["id"], str(e)[:120]))
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}  {nbytes/1e6:.0f} MB  {time.time()-t0:.0f}s  fails={len(fails)}", flush=True)
    print(f"finished: ok={len(todo)-len(fails)} fail={len(fails)} {nbytes/1e6:.0f} MB in {time.time()-t0:.0f}s")
    for f in fails[:20]:
        print("FAIL", f)


if __name__ == "__main__":
    main()
