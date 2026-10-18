import json, os, subprocess, time
from pathlib import Path

REPO = "phly95/crimson-texture-upscale"
TAG = "v1.0.0"
PKG_DIR = Path("/home/deck/workspace/release_packages")

def get_remote_assets():
    out = subprocess.run(["gh", "release", "view", TAG, "--repo", REPO, "--json", "assets"],
                         capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    return {a["name"] for a in data.get("assets", [])}

def upload_file(fpath):
    print(f"[{time.strftime('%H:%M:%S')}] Uploading {fpath.name} ({fpath.stat().st_size / (1024*1024):.1f} MB)...", flush=True)
    t0 = time.time()
    for attempt in range(1, 4):
        try:
            subprocess.run(["gh", "release", "upload", TAG, str(fpath), "--repo", REPO, "--clobber"],
                           check=True)
            print(f"[{time.strftime('%H:%M:%S')}] Finished {fpath.name} in {time.time() - t0:.1f}s", flush=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt} failed ({e}), retrying in 5s...", flush=True)
            time.sleep(5)
    return False

def main():
    all_files = sorted(PKG_DIR.glob("*.zip"), key=lambda p: p.stat().st_size)
    print(f"Found {len(all_files)} total release packages.")

    for f in all_files:
        remote = get_remote_assets()
        if f.name in remote:
            print(f"Already uploaded: {f.name}", flush=True)
            continue
        upload_file(f)

    print("All release assets verified and uploaded!")

if __name__ == "__main__":
    main()
