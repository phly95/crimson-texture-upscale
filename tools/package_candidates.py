import os, sys, subprocess, time
from pathlib import Path

ROOT = Path("/home/deck/workspace/csvm-hd")
PIPE_UNITS = ROOT / "pipeline" / "units"
OUT_DIR = Path("/home/deck/workspace/release_packages")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_dir_size(d):
    sz = 0
    for root, dirs, files in os.walk(d):
        for f in files:
            sz += os.path.getsize(os.path.join(root, f))
    return sz

def bucket_units(u_list, max_mb=1700):
    buckets = []
    cur_bucket = []
    cur_size = 0
    for u in u_list:
        sz = get_dir_size(PIPE_UNITS / u)
        if cur_size + sz > max_mb * 1024 * 1024 and cur_bucket:
            buckets.append((cur_bucket, cur_size))
            cur_bucket = [u]
            cur_size = sz
        else:
            cur_bucket.append(u)
            cur_size += sz
    if cur_bucket:
        buckets.append((cur_bucket, cur_size))
    return buckets

def make_archive(archive_name, units_list):
    out_zip = OUT_DIR / f"{archive_name}.zip"
    if out_zip.exists():
        print(f"Skipping {out_zip.name} (already exists, {out_zip.stat().st_size / (1024*1024):.1f} MB)")
        return out_zip
    
    print(f"Creating {out_zip.name} with {len(units_list)} units...")
    t0 = time.time()
    
    # Write list of unit relative paths
    list_file = OUT_DIR / f"{archive_name}_filelist.txt"
    with open(list_file, "w") as f:
        for u in units_list:
            f.write(f"{u}\n")
            
    # Run zip -0 from pipeline/units directory
    cmd = ["zip", "-0", "-q", "-r", str(out_zip), "-@", "<", str(list_file)]
    # Use shell pipe
    cmd_str = f"cd {PIPE_UNITS} && zip -0 -q -r {out_zip} . -i " + " ".join([f"'{u}/*'" for u in units_list])
    # For large unit list, passing file list via -@ is cleaner
    res = subprocess.run(f"cd {PIPE_UNITS} && zip -0 -q -r {out_zip} -@ < {list_file}", shell=True, check=True)
    list_file.unlink(missing_ok=True)
    
    sz_mb = out_zip.stat().st_size / (1024*1024)
    print(f"  Finished {out_zip.name}: {sz_mb:.1f} MB in {time.time() - t0:.1f}s")
    return out_zip

def main():
    all_units = sorted(os.listdir(PIPE_UNITS))
    world_units = [u for u in all_units if not (u.startswith("rimage_") or u.startswith("gfx_") or u.startswith("scrap_"))]
    ui_rimage = [u for u in all_units if u.startswith("rimage_")]
    ui_gfx = [u for u in all_units if u.startswith("gfx_")]
    ui_scrap = [u for u in all_units if u.startswith("scrap_")]
    
    tasks = []
    
    # 1. World
    w_buckets = bucket_units(world_units, 1700)
    for i, (b, sz) in enumerate(w_buckets, 1):
        tasks.append((f"candidates_world_part{i}", b))
        
    # 2. UI Rimage
    r_buckets = bucket_units(ui_rimage, 1700)
    for i, (b, sz) in enumerate(r_buckets, 1):
        tasks.append((f"candidates_ui_rimage_part{i}", b))
        
    # 3. UI Gfx
    g_buckets = bucket_units(ui_gfx, 1700)
    for i, (b, sz) in enumerate(g_buckets, 1):
        tasks.append((f"candidates_ui_gfx_part{i}", b))
        
    # 4. UI Scrap
    s_buckets = bucket_units(ui_scrap, 1700)
    for i, (b, sz) in enumerate(s_buckets, 1):
        tasks.append((f"candidates_ui_scrap_part{i}", b))
        
    print(f"Total packages to create: {len(tasks)}")
    for name, ulist in tasks:
        make_archive(name, ulist)
        
    print("\nAll archives packaged in", OUT_DIR)

if __name__ == "__main__":
    main()
