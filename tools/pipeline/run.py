#!/usr/bin/env python3
"""run.py - the long-running CSVM prompt + redraw pipeline (Qwen3.8-Flash-Next writes prompts, FLUX.2 klein 9B redraws).

    venv/bin/python -I tools/pipeline/run.py            # runs until every unit is done (resumable; Ctrl-C or STOP file ends it cleanly)

It works in batches (config.json "batch", default 10): the writer phase has Strata (Flash-Next, CPU image encoder) write a
prompt for each unit in the batch, then Strata is stopped and klein 9B redraws every prompted unit once per seed; then it
switches back.  A switch costs about 33 s (Strata start 28 s + first request ~4 s + stop 1 s), about 3% of a batch.

Files, all under pipeline/:
  config.json        read again at the start of every batch, so it can be edited while the run is going
  state.json         what it is doing right now (the live page reads it)
  index.json         one light record per touched unit, newest first (the live page reads it)
  units/<uid>/       in.png orig.png hint.txt digest.txt prompt.txt klein_s<seed>.png meta.json logs
  STOP               touch it: the current batch is finished (prompted units still get their redraw) and the run exits
  PAUSE              touch it: waits (between units) until the file is removed
  run.log            plain log
Nothing here touches the game install or the texture pack.
"""
import json, os, random, re, shutil, signal, subprocess, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/home/deck/workspace/csvm-hd")
PIPE = Path(os.environ.get("PIPE_DIR", ROOT / "pipeline"))
UN = PIPE / "units"
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "tools" / "qwen"))
sys.path.insert(0, str(ROOT / "tools" / "pipeline"))
import prep_unit                                   # noqa: E402
from colormatch import colormatch
from finalize import finalize                  # noqa: E402
from magentakey import magenta_key             # noqa: E402,F401
from orient import detect_rotation                 # noqa: E402
from analyze_tex import analyze                    # noqa: E402

PY = str(ROOT / "venv" / "bin" / "python")
KLEIN = str(ROOT / "tools" / "qwen" / "klein.sh")
import klein_server as kserver                      # noqa: E402  sd-server with klein kept loaded (cloud mode); sd-cli per image otherwise
SERVER_ON = {"v": False, "fails": 0}
PREFETCH = {}                                   # (input path, seed) -> {"jid", "prompt"}: klein jobs already queued on the server for upcoming units
WRITER = str(ROOT / "tools" / "qwen" / "vlm_prompt.py")
REVIEWER = str(ROOT / "tools" / "qwen" / "vlm_review.py")
STRATA_CFG = "strata-iq3_s-vision.json"
DEFAULTS = {"batch": 10, "seeds": [1, 2], "order": "shuffle", "order_seed": 1, "skip_flipbooks": True,
            "priority": ["25bigstink__3f364ab5", "fhunter_logo2__cbba6a35", "crop04__4818f512", "palm2__dd3d0665", "turb02__3304c2f0"],
            "max_units": None, "writer_tries": 2, "klein_tries": 2}

stop_flag = {"v": False}


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(PIPE / "run.log", "a") as f:
        f.write(line + "\n")


def atomic(path, obj):
    tmp = Path(f"{path}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def load_cfg():
    cfg = dict(DEFAULTS)
    p = PIPE / "config.json"
    if p.exists():
        try:
            cfg.update(json.load(open(p)))
        except Exception as e:                                 # noqa: BLE001
            log(f"config.json unreadable ({e}); using the last good values/defaults")
    else:
        atomic(p, DEFAULTS)
    return cfg


# ------------------------------------------------------------------ state
STATE = {"phase": "starting", "detail": "", "started": time.time(), "run_done": 0, "batch_no": 0, "batch_units": [],
         "current": None, "unit_started": None, "last_error": None, "recent": []}
INDEX = {}                                                     # uid -> light record


_PS_LOCK = threading.Lock()


def cloud_cfg(cfg):
    """config "cloud_writer": {url, model, key_file, extra, parallel}: prompts come from a cloud model (many in parallel, Strata never starts, klein keeps the GPU)"""
    c = cfg.get("cloud_writer")
    return c if c and c.get("enabled", True) else None


def writer_env(cfg):
    c = cloud_cfg(cfg)
    if not c:
        return dict(os.environ)
    return dict(os.environ, VLM_URL=c["url"], VLM_MODEL=c["model"], VLM_EXTRA=json.dumps(c.get("extra", {})), VLM_KEY_FILE=str(ROOT / c.get("key_file", "gemini_api.key")))


def push_state(**kw):
    with _PS_LOCK:
        _push_state(**kw)


def _push_state(**kw):
    STATE.update(kw)
    STATE["updated"] = time.time()
    el = time.time() - STATE["started"]
    STATE["elapsed"] = el
    if STATE["run_done"]:
        per = el / STATE["run_done"]
        STATE["sec_per_unit"] = per
        STATE["eta_s"] = per * max(0, STATE.get("pending", 0))
    atomic(PIPE / "state.json", STATE)


def push_index():
    recs = sorted(INDEX.values(), key=lambda r: -r["t"])
    atomic(PIPE / "index.json", {"units": recs, "updated": time.time()})


def meta_path(uid):
    return UN / uid / "meta.json"


def read_meta(uid):
    try:
        return json.load(open(meta_path(uid)))
    except Exception:                                          # noqa: BLE001
        return None


def save_meta(m):
    atomic(meta_path(m["uid"]), m)
    INDEX[m["uid"]] = {"uid": m["uid"], "stem": m["stem"], "status": m["status"], "w": m["size"][0], "h": m["size"][1],
                       "cutout": m["cutout"], "t": m.get("t_update", time.time()), "batch": m.get("batch"),
                       "seeds": sorted(int(s) for s in m.get("klein", {}) if m["klein"][s].get("ok")),
                       "writer_s": (m.get("writer") or {}).get("s"), "family": m.get("family"), "error": m.get("error")}
    push_index()


# ------------------------------------------------------------------ units
def enforce_original_rule(cfg, idx):
    """config 'original_min_dim' (800): a texture with a side that long keeps its original (full-screen art is not worth redrawing).  Appends an 'original'
    decision for every such unit whose latest decision is not already 'original' (approvals included) and returns the set of those uids."""
    lim = cfg.get("original_min_dim")
    big = {uid for uid, (d, u) in idx.items() if lim and max(u.get("size") or [0, 0]) >= lim}
    big |= {u for u in (cfg.get("original_uids") or []) if u in idx}          # explicit list: plain-colour pictures etc.
    if not big:
        return set()
    rp = PIPE / "reviews.jsonl"
    last = {}
    if rp.exists():
        for ln in rp.read_text().splitlines():
            try:
                r = json.loads(ln); last[r["uid"]] = r
            except (ValueError, KeyError):
                continue
    new = [u for u in sorted(big) if last.get(u, {}).get("decision") != "original"]
    if new:
        with open(rp, "a") as f:
            for u in new:
                f.write(json.dumps({"uid": u, "decision": "original", "chosen": last.get(u, {}).get("chosen"), "seen": [], "issues": [],
                                    "text": f"auto: uses the original (a side is {lim} px or more, or listed in original_uids)", "ts": time.time()}) + "\n")
        log(f"original rule ({lim} px): {len(new)} units set to use the original")
    return big


def universe(cfg):
    prep_unit.find("x") if False else None
    try:
        prep_unit.find("__none__")
    except KeyError:
        pass
    idx = prep_unit._idx
    tasks = {t["id"]: t for t in json.load(open(ROOT / "tasks" / "tasks.json"))}
    uids = []
    keep_orig = enforce_original_rule(cfg, idx)
    for uid, (d, u) in idx.items():
        if uid in keep_orig:
            continue
        if cfg["skip_flipbooks"] and u.get("flipbook_id"):
            continue
        uids.append(uid)
    uids.sort()
    if cfg["order"] == "shuffle":
        random.Random(cfg["order_seed"]).shuffle(uids)
    pri = [u for u in cfg["priority"] if u in idx and u in uids]
    uids = pri + [u for u in uids if u not in pri]
    return uids, idx, tasks


def dark_gamma(orig_path):
    """very dark pictures (night scenes, deep shadow): the writer and klein cannot see structure at RGB 3-30, so they get a brightened copy
    (gamma g < 1, median of the opaque pixels -> ~0.3 at the 75th percentile) and the redraw is darkened back with gamma 1/g. None = not dark."""
    import numpy as np
    from PIL import Image
    a4 = np.asarray(Image.open(orig_path).convert("RGBA")).astype(np.float32) / 255
    m = a4[..., 3] > 0.5
    if m.sum() < 50:
        return None
    L = (a4[..., :3] @ np.array([.299, .587, .114], np.float32))[m]
    if float(np.percentile(L, 50)) > 0.07:
        return None
    return float(np.clip(np.log(0.30) / np.log(max(float(np.percentile(L, 75)), 0.01)), 0.3, 1.0))


def apply_gamma(img, g):
    import numpy as np
    from PIL import Image
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255
    return Image.fromarray((np.clip(a ** g, 0, 1) * 255 + 0.5).astype(np.uint8))


def sibling_context(uid, stem):
    """other textures of the same set (same file-name prefix, e.g. steinmann03/04/09/12): what the finished ones were read as, so a bare ground texture
    is not guessed in isolation (a dark blue-grey ground in an industrial set is deck/concrete, not moss)"""
    pre = re.sub(r"[\d_]+$", "", stem)
    if len(pre) < 5:
        return ""
    out = []
    for u, r in sorted(INDEX.items()):
        if u == uid or r.get("status") != "done" or not str(r.get("stem", "")).startswith(pre) or re.sub(r"[\d_]+$", "", str(r.get("stem"))) != pre:
            continue
        try:
            t = (UN / u / "prompt.txt").read_text()
            pr = re.search(r"PROMPT:\s*(.+)", t, re.S).group(1).strip()
            first = re.split(r"(?<=[.!?])\s", pr)[0]                # the redraw prompt's first sentence says what the picture is
            first = re.sub(r"^(Redraw|Recreate|Render)\s+(this\s+)?(as\s+)?(an?\s+)?", "", first)
            first = re.split(r",\s*(?:filling|keeping|with no|no |not |on a (?:flat|plain))", first)[0]       # cut the boilerplate that follows the description
            first = re.sub(r"\b(crisp|high-resolution|realistic),?\s*", "", first)
            first = re.sub(r"\s+on a (?:flat |plain )?(?:pure )?green.*$", "", first)                 # the chroma-key background is not part of what the picture is
            out.append(f"{r['stem']} = {first.strip(' ,.')[:170]}")
        except Exception:                                      # noqa: BLE001
            continue
        if len(out) >= 4:
            break
    if not out:
        return ""
    return (f"; SET CONTEXT: other textures of the same set (file-name prefix '{pre}'): " + " | ".join(out)
            + ". They share a theme and material family: use this as strong evidence for what this texture is, and do not invent materials the set does not suggest (for example vegetation or soil in an industrial set)")


def new_meta(uid, idx, tasks, batch):
    d, u = idx[uid]
    inp, orig, hint = prep_unit.prep(uid, UN / uid)
    for src, dst in ((f"{uid}_in.png", "in.png"), (f"{uid}_orig.png", "orig.png"), (f"{uid}_hint.txt", "hint.txt")):
        os.replace(UN / uid / src, UN / uid / dst)
    from PIL import Image
    rot, rwords = 0, []
    try:
        rot, oinfo = detect_rotation(Image.open(UN / uid / "orig.png"))
        rwords = (oinfo.get("strong_words") or {}).get(rot, [])
    except Exception as e:                                     # noqa: BLE001
        log(f"{uid}: orientation check failed: {e!r}")
    if rot:                                                    # lettering is rotated: the writer and klein get the upright picture, the result is rotated back later
        d0 = UN / uid
        Image.open(d0 / "orig.png").convert("RGBA").rotate(-rot, expand=True).save(d0 / "orig_up.png")
        Image.open(d0 / "in.png").rotate(-rot, expand=True).save(d0 / "in.png")
        uw, uh = Image.open(d0 / "orig_up.png").size
        hint = re.sub(r"\d+x\d+ px", f"{uw}x{uh} px", hint, count=1) + f"; the lettering ('{' '.join(rwords)}') was rotated {rot} degrees clockwise to read upright, and this is the upright version"
        (d0 / "hint.txt").write_text(hint)
        log(f"{uid}: lettering {rwords} found rotated; processing upright (rotated {rot} deg clockwise), result will be rotated back")
    lift = None
    try:
        d0 = UN / uid
        src_o = d0 / ("orig_up.png" if rot else "orig.png")
        lift = dark_gamma(src_o)
        if lift is not None:                                   # brighten the writer's picture, the measurements and klein's input; redraw is darkened back later
            Image.open(d0 / "in.png").save(d0 / "in_dark.png")
            apply_gamma(Image.open(d0 / "in.png"), lift).save(d0 / "in.png")
            o4 = Image.open(src_o).convert("RGBA")
            lo = apply_gamma(o4, lift); lo.putalpha(o4.getchannel("A")); lo.save(d0 / "orig_lift.png")
            hint += "; this picture is very dark, so the copy shown to you and all measurements are brightened (gamma %.2f) to make structures visible: describe every structure you can see (buildings, trees, roads, lights); the result is darkened back afterwards" % lift
            (d0 / "hint.txt").write_text(hint)
            log(f"{uid}: very dark picture, brightened with gamma {lift:.2f} for processing")
    except Exception as e:                                     # noqa: BLE001
        lift = None
        log(f"{uid}: dark-picture check failed: {e!r}")
    try:
        (UN / uid / "digest.txt").write_text(analyze(str(UN / uid / ("orig_lift.png" if lift is not None else "orig_up.png" if rot else "orig.png")))[0])
    except Exception as e:                                     # noqa: BLE001
        (UN / uid / "digest.txt").write_text(f"analysis failed: {e}")
    cut0 = "cut-out" in hint                                   # decided from the original hint: reviewer notes and earlier prompts may say "cut-out" too
    try:
        fbh = feedback_hint(uid)
        if fbh:
            hint += fbh
            (UN / uid / "hint.txt").write_text(hint)
    except Exception as e:                                     # noqa: BLE001
        log(f"{uid}: feedback hint failed: {e!r}")
    try:
        sib = sibling_context(uid, u["stem"])
        if sib:
            hint += sib
            (UN / uid / "hint.txt").write_text(hint)
    except Exception as e:                                     # noqa: BLE001
        log(f"{uid}: sibling context failed: {e!r}")
    w, h = Image.open(UN / uid / "orig.png").size
    cut = cut0
    tk = d.name
    return {"uid": uid, "stem": u["stem"], "task": tk, "family": tasks.get(tk, {}).get("family"), "size": [w, h],
            "target": u.get("target"), "cutout": bool(cut), "alpha_class": u.get("alpha_class"), "tileable": u.get("tileable_guess"),
            "seam_ratio": u.get("seam_ratio"), "look": u.get("look"), "chapters": u.get("chapters"),
            "nodes": ((u.get("usage") or {}).get("nodes") or [])[:6], "hint": hint, "status": "pending", "klein": {},
            "batch": batch, "rot": rot, "rot_words": rwords, "lift": lift, "t_update": time.time()}


# ------------------------------------------------------------------ Strata / GPU
def vram_mib():
    try:
        return int(open("/sys/class/drm/card0/device/mem_info_vram_used").read()) // 1048576
    except Exception:                                          # noqa: BLE001
        return 0


def gtt_mib():
    try:
        return int(open("/sys/class/drm/card0/device/mem_info_gtt_used").read()) // 1048576
    except Exception:                                          # noqa: BLE001
        return 0


def strata_up():
    try:
        urllib.request.urlopen("http://127.0.0.1:9931/v1/models", timeout=3).read()
        return True
    except Exception:                                          # noqa: BLE001
        return False


def start_strata():
    kserver.stop()                                             # Strata and the loaded klein server cannot share the GPU
    if strata_up():
        return
    push_state(phase="switching", detail="starting Strata (Qwen3.8-Flash-Next)")
    t = time.time()
    subprocess.run(["bash", "-c", f"source ~/AppData/Local_LLM/llm-aliases.sh; _llm_bg {STRATA_CFG}"], capture_output=True, timeout=600)
    for _ in range(120):
        if strata_up():
            break
        time.sleep(2)
    else:
        raise RuntimeError("Strata did not come up")
    log(f"Strata ready in {time.time()-t:.0f}s")


def stop_strata():
    push_state(phase="switching", detail="stopping Strata, freeing the GPU")
    subprocess.run(["podman", "stop", "llm-strata"], capture_output=True, timeout=120)
    t = time.time()
    for _ in range(180):
        if vram_mib() < 1000 and gtt_mib() < 2500:             # Vulkan pinned host memory is capped by the 32 GB GTT window, which Strata's pinned RAM holds until released
            time.sleep(3)
            log(f"GPU free after {time.time()-t:.0f}s (VRAM {vram_mib()} MiB, GTT {gtt_mib()} MiB)")
            return
        time.sleep(1)
    log(f"warning: after stopping Strata VRAM is {vram_mib()} MiB, GTT {gtt_mib()} MiB")


# ------------------------------------------------------------------ phases
def klein_run(ref, out, prompt, W, H, seed, outlog, errlog, timeout=150):
    """one klein redraw: through the loaded sd-server in cloud mode (about 5 s per image faster, bit-identical output), else a fresh sd-cli (klein.sh).
    Any server trouble stops the server (frees the VRAM) and this image is drawn by sd-cli; three such failures switch the server off for the run."""
    if SERVER_ON["v"]:
        try:
            if not kserver.up():
                PREFETCH.clear()
                kserver.start(PIPE / "klein_server.log")
            elif kserver.COUNT["n"] >= 400 and not PREFETCH:   # a precaution against slow leaks in a long-lived server
                kserver.stop(); kserver.start(PIPE / "klein_server.log")
            t0 = time.time()
            pre = PREFETCH.pop((str(ref), int(seed)), None)
            if pre is not None and pre["prompt"] == prompt:    # already queued (and probably already drawn) while the previous unit was post-processed
                kserver.collect(pre["jid"], out, timeout=timeout + 120, tick=push_state)
            else:
                kserver.gen(ref, out, prompt, W, H, seed, timeout=timeout, tick=push_state)
            Path(outlog).write_text(f"klein server: {time.time() - t0:.1f} s\n")
            return 0
        except Exception as e:                                 # noqa: BLE001
            SERVER_ON["fails"] += 1
            PREFETCH.clear()
            log(f"klein server failed ({e!r}); stopping it and drawing this image with sd-cli" + ("; server switched off for this run" if SERVER_ON["fails"] >= 3 else ""))
            kserver.stop()
            if SERVER_ON["fails"] >= 3:
                SERVER_ON["v"] = False
    env = dict(os.environ, W=str(W), H=str(H), SEED=str(seed))
    return heartbeat_run([KLEIN, "9b", str(ref), str(out), prompt], env, outlog, errlog, timeout=timeout, fail_on="Failed to allocate pinned memory")


def prefetch_unit(m, cfg):
    """cloud mode: queue the unit's klein jobs on the server now, while the previous unit is still being post-processed, so the GPU never waits"""
    if not SERVER_ON["v"] or len(PREFETCH) >= 6:
        return
    d = UN / m["uid"]
    try:
        if not kserver.up():
            return                                             # klein_run starts it
        from PIL import Image
        prompt = re.search(r"PROMPT:\s*(.+)", (d / "prompt.txt").read_text(), re.S).group(1).strip()
        W, H = Image.open(d / "in.png").size
        for seed in cfg["seeds"]:
            key = (str(d / "in.png"), int(seed))
            if key not in PREFETCH:
                PREFETCH[key] = {"jid": kserver.submit(d / "in.png", prompt, W, H, seed), "prompt": prompt}
    except Exception as e:                                     # noqa: BLE001
        log(f"prefetch for {m['uid']} failed ({e!r}); not prefetching this one")


def heartbeat_run(cmd, env, out_path=None, err_path=None, timeout=1500, fail_on=None):
    """Popen with a state heartbeat every 5 s so the page shows the run is alive."""
    t = time.time()
    with open(out_path or os.devnull, "w") as fo, open(err_path or os.devnull, "w") as fe:
        p = subprocess.Popen(cmd, stdout=fo, stderr=fe, env=env)
        while p.poll() is None:
            time.sleep(5)
            push_state()
            if fail_on and out_path and Path(out_path).exists() and fail_on in Path(out_path).read_text(errors="ignore"):
                log(f"{cmd[0].split('/')[-1]}: '{fail_on}' in its log (GTT {gtt_mib()} MiB, VRAM {vram_mib()} MiB); killing the attempt early")
                p.kill(); p.wait()
                return 125
            if time.time() - t > timeout:
                try:
                    st = subprocess.run(["ps", "-o", "stat=,wchan:20=", "-p", str(p.pid)], capture_output=True, text=True).stdout.strip()
                except Exception:                              # noqa: BLE001
                    st = "?"
                log(f"TIMEOUT after {timeout}s: {cmd[0].split('/')[-1]} pid {p.pid} state/wchan [{st}], VRAM {vram_mib()} MiB, Strata up: {strata_up()}")
                p.kill(); p.wait()
                return 124
    return p.returncode


def write_prompt(m, cfg):
    uid = m["uid"]; d = UN / uid
    for attempt in range(1, cfg["writer_tries"] + 1):
        t = time.time()
        rc = heartbeat_run([PY, "-I", WRITER, str(d / "in.png"), str(d / ("orig_lift.png" if (d / "orig_lift.png").exists() else "orig_up.png" if (d / "orig_up.png").exists() else "orig.png")), m["hint"]], writer_env(cfg), d / "prompt.txt", d / "writer.err", timeout=(900 if cloud_cfg(cfg) else 1500))
        txt = (d / "prompt.txt").read_text() if (d / "prompt.txt").exists() else ""
        mm = re.search(r"PROMPT:\s*(.+)", txt, re.S)
        toks = 0
        for ln in (d / "writer.err").read_text().splitlines():
            if "completion_tokens" in ln:
                try:
                    toks = int(ln.split("'completion_tokens': ")[1].split(",")[0])
                except Exception:                              # noqa: BLE001
                    pass
        if rc == 0 and mm and len(mm.group(1).split()) >= 40:
            m["writer"] = {"s": round(time.time() - t, 1), "tokens": toks, "tries": attempt, **({"model": cloud_cfg(cfg)["model"]} if cloud_cfg(cfg) else {})}
            m["status"] = "prompted"; m["pv"] = PV; m["t_update"] = time.time()
            return True
        log(f"{uid}: writer attempt {attempt} failed (rc {rc})")
        if cloud_cfg(cfg):
            time.sleep(15)
        elif not strata_up():
            start_strata()
    if cloud_cfg(cfg):                                         # network or quota trouble must not fail a unit for good: try again in a later batch (3 batches at most)
        m["cloud_fails"] = m.get("cloud_fails", 0) + 1
        if m["cloud_fails"] < 3:
            m["status"] = "pending"; m["t_update"] = time.time()
            return False
    m["status"] = "failed"; m["error"] = "writer produced no usable prompt"; m["t_update"] = time.time()
    return False


SOFT_SKIP_FILE = PIPE / "soft_skip.json"
SOFT_SKIP = set()


def load_soft_skip():
    try:
        SOFT_SKIP.update(json.loads(SOFT_SKIP_FILE.read_text()))
    except Exception:                                          # noqa: BLE001
        pass


def save_soft_skip():
    atomic(SOFT_SKIP_FILE, sorted(SOFT_SKIP))


FEEDBACK_FILE = PIPE / "feedback.json"
ISSUE_TEXT = {
    "lettering": "the lettering was wrong (misspelled, doubled or missing letters)",
    "stray": "extra text, numbers or symbols were painted on the picture that do not belong there",
    "added": "things were added that are not in the original (objects, nails, texture, creatures)",
    "missing": "details or objects of the original are missing",
    "shape": "the shape, pose or layout differs from the original",
    "colour": "the colours differ from the original (keep the original's colours, brightness and tint)",
    "material": "it looks fake or like the wrong material",
    "cartoon": "it looks too cartoonish or painted; it must look as photographic or realistic as the original does",
    "blurry": "it is blurry or soft",
    "edges": "the cut-out edges are bad or have a green fringe",
    "tiling": "the texture is repeated over and over in the game and the redraw did not tile: opposite edges did not match (visible seams), or it had a vignette, border, gradient, a large feature or an object cut off at an edge. Make it a flat, evenly lit, edge-to-edge pattern of uniform density in which nothing large touches or crosses the edges and every edge continues into the opposite edge",
    "other": "something else is wrong (see the reviewer's note)",
}


def load_feedback():
    try:
        return json.loads(FEEDBACK_FILE.read_text())
    except Exception:                                          # noqa: BLE001
        return {}


def feedback_hint(uid):
    f = load_feedback().get(uid)
    if not f or f.get("skipped"):
        return ""
    probs = [ISSUE_TEXT[i] for i in f.get("issues", []) if i in ISSUE_TEXT]
    out = "; HUMAN REVIEW OF AN EARLIER REDRAW OF THIS TEXTURE (it was rejected): problems: " + ("; ".join(probs) or "not specified")
    if f.get("text"):
        out += f". Reviewer's note: \"{f['text']}\""
    if f.get("prev"):
        prev = re.sub(r",?\s*\b[A-Z](?:-[A-Z]){2,}\b,?", "", f["prev"])      # letter-by-letter spellings made the editor paint hyphens
        out += ". The earlier prompt was: " + prev[:900]
    return out + ". Write a new prompt that fixes exactly these problems and keeps what was right"


def process_reject_feedback():
    """thumbs-down on the review page ('It is used to redo this one'): archive that result and redo it with the reviewer's issues and note given to the writer.
    Rejections that are only about alpha (finalize handles that) are recorded but not redone; tiling rejections get a regen whose prompt asks for an even, edge-to-edge pattern."""
    rp = PIPE / "reviews.jsonl"
    if not rp.exists():
        return
    last = {}
    for ln in rp.read_text().splitlines():
        try:
            r = json.loads(ln); last[r["uid"]] = r
        except (ValueError, KeyError):
            continue
    fb = load_feedback(); changed = False
    for ruid, r in last.items():
        if r.get("decision") != "reject" or r.get("route") == "agent" or (ruid in fb and fb[ruid].get("ts", 0) >= r["ts"]):
            continue                                           # route == agent: a drawing agent redoes it (tools/agents/dispatch_feedback.py)
        m0 = read_meta(ruid)
        issues = list(r.get("issues") or [])
        entry = {"issues": issues, "text": r.get("text") or "", "ts": r["ts"]}
        actionable = [i for i in issues if i != "alpha"] or bool(entry["text"])
        if not m0 or m0.get("status") != "done" or m0.get("t_update", 0) > r["ts"] or not actionable:
            entry["skipped"] = True                            # redone since, or nothing a prompt can fix
            fb[ruid] = entry; changed = True
            continue
        try:
            entry["prev"] = re.search(r"PROMPT:\s*(.+)", (UN / ruid / "prompt.txt").read_text(), re.S).group(1).strip()
        except Exception:                                      # noqa: BLE001
            entry["prev"] = ""
        fb[ruid] = entry; changed = True
        dst = PIPE / "archive" / "feedback" / f"{ruid}_{int(r['ts'])}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.move(str(UN / ruid), str(dst))
        INDEX.pop(ruid, None)
        log(f"{ruid}: rejected in review ({', '.join(issues) or 'note only'}); archived to {dst}, will be redone with the feedback")
    if changed:
        atomic(FEEDBACK_FILE, fb)


def process_redraw_requests():
    """review page 'Could be redrawn' on a smooth-enlargement unit: archive that result and send the unit through the normal AI redraw (never the soft route again)"""
    rp = PIPE / "reviews.jsonl"
    if not rp.exists():
        return
    last = {}
    for ln in rp.read_text().splitlines():
        try:
            r = json.loads(ln); last[r["uid"]] = r
        except (ValueError, KeyError):
            continue
    for ruid, r in last.items():
        if r.get("decision") != "redraw" or ruid in SOFT_SKIP:
            continue
        SOFT_SKIP.add(ruid); save_soft_skip()
        m0 = read_meta(ruid)
        if m0 and m0.get("soft"):
            dst = PIPE / "archive" / "redraw" / ruid
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(UN / ruid), str(dst))
            INDEX.pop(ruid, None)
            log(f"{ruid}: redraw requested from the review page; smooth enlargement archived to {dst}")


def is_soft_effect(orig_path):
    """shadows, smoke, glows, flares, tracers: almost all alpha is semi-transparent, no opaque core, smooth alpha.  klein cannot paint soft alpha (it makes hard blobs and
    green fringes), and there is nothing to redraw in a smooth gradient, so these get a clean smooth enlargement that keeps the original alpha."""
    import numpy as np, cv2
    from PIL import Image
    a = np.asarray(Image.open(orig_path).convert("RGBA").getchannel("A")).astype(np.float32)
    nz = a > 8
    if nz.sum() < 10:
        return False
    soft = ((a > 8) & (a < 240)).sum() / nz.sum()
    op = (a >= 240).sum() / nz.sum()
    g = np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)) / 8
    return bool(soft >= 0.85 and op <= 0.15 and float(np.percentile(g[nz], 90)) <= 50)


def render_soft(m):
    """premultiplied bicubic enlargement (x8, long side <= 2048) of the RGBA original with a little dither against banding in big soft gradients"""
    import numpy as np, cv2
    from PIL import Image
    uid = m["uid"]; d = UN / uid
    o = np.asarray(Image.open(d / "orig.png").convert("RGBA")).astype(np.float32) / 255
    h, w = o.shape[:2]
    W, H = min(w * 8, 2048), min(h * 8, 2048)
    pm = np.dstack([o[..., :3] * o[..., 3:4], o[..., 3:4]])
    up = cv2.resize(pm, (W, H), interpolation=cv2.INTER_CUBIC)
    a = np.clip(up[..., 3:4], 0, 1)
    rgb = np.clip(up[..., :3] / np.maximum(a, 1e-4), 0, 1)
    rng = np.random.default_rng(1)
    noise = (rng.random((H, W, 1), np.float32) + rng.random((H, W, 1), np.float32) - 1) / 255      # triangular dither, +-1 level
    rgb = np.clip(rgb + noise, 0, 1)
    a = np.clip(a + noise, 0, 1)
    Image.fromarray((np.dstack([rgb, a]) * 255 + 0.5).astype(np.uint8), "RGBA").save(d / "klein_s1.png")
    m["klein"] = {"1": {"ok": True, "s": 0.0, "soft": True, "cm": False, "q": {"flags": [], "score": 9.0}}}
    m["best"] = "1"; m["soft"] = True; m["status"] = "done"; m["pv"] = PV; m["t_update"] = time.time()


def _score(v):
    return ((v or {}).get("q_cm") or (v or {}).get("q") or {}).get("score", -9)


def needs_review(m):
    """cheap, high-recall triggers; the VLM review decides.  Lettering was requested, or every redraw lost the silhouette."""
    try:
        import textcheck
        pr = re.search(r"PROMPT:\s*(.+)", (UN / m["uid"] / "prompt.txt").read_text(), re.S).group(1)
        if textcheck.expected_words(pr):
            return True
    except Exception:                                          # noqa: BLE001
        return False
    ks = [v for v in (m.get("klein") or {}).values() if v.get("ok")]
    return bool(ks) and all("shape drift" in (((v.get("q_cm") or v.get("q") or {}).get("flags")) or []) for v in ks)


def review_unit(m, cfg):
    """Flash-Next looks at the original and the redraw, says what went wrong and (if clearly fixable) writes a new prompt -> retry/prompt.txt"""
    uid = m["uid"]; d = UN / uid
    seed = m.get("best") or next(iter(m["klein"]))
    t = time.time()
    rc = heartbeat_run([PY, "-I", REVIEWER, str(d), str(seed)], writer_env(cfg), d / "review.txt", d / "review.err", timeout=900)
    txt = (d / "review.txt").read_text() if (d / "review.txt").exists() else ""
    rv = re.search(r"REVIEW:\s*(.*?)\s*VERDICT:", txt, re.S); vd = re.search(r"VERDICT:\s*(OK|REDO)", txt, re.I); pm = re.search(r"PROMPT:\s*(.+)", txt, re.S)
    rec = {"t": time.time(), "s": round(time.time() - t, 1), "seed": str(seed), "text": (rv.group(1).strip() if rv else txt[:600]), "verdict": (vd.group(1).lower() if vd else "failed")}
    if rc == 0 and rec["verdict"] == "redo":
        if pm and len(pm.group(1).split()) >= 40:
            (d / "retry").mkdir(exist_ok=True)
            (d / "retry" / "prompt.txt").write_text("PROMPT: " + pm.group(1).strip())
            m["retry"] = {"status": "pending"}
        else:
            rec["verdict"] = "failed"
    elif rc != 0:
        rec["verdict"] = "failed"
    m["review"] = rec
    log(f"{uid}: review {rec['verdict']} ({rec['s']}s): {rec['text'][:140]}")


def render_retry(m, cfg):
    """second klein try with the corrected prompt; results in unit/retry/; the better of the two becomes m['final']"""
    uid = m["uid"]; d = UN / uid; r = d / "retry"
    from PIL import Image
    prompt = (r / "prompt.txt").read_text().replace("PROMPT:", "", 1).strip()
    shutil.copy(d / "orig.png", r / "orig.png")
    W, H = Image.open(d / "in.png").size
    recs = {}
    for seed in cfg["seeds"]:
        out = r / f"klein_s{seed}.png"
        for attempt in range(1, cfg["klein_tries"] + 1):
            t = time.time()
            rc = klein_run(d / "in.png", out, prompt, W, H, seed, r / f"klein_s{seed}_a{attempt}.log", r / f"klein_s{seed}_a{attempt}.err")
            if rc == 0 and out.exists() and out.stat().st_size > 1000:
                if m.get("lift"):
                    kl_b = Image.open(out); kl_b.save(r / f"klein_s{seed}_lifted.png")
                    apply_gamma(kl_b, 1.0 / m["lift"]).save(out)
                if m.get("rot"):
                    img_up = Image.open(out); img_up.save(r / f"klein_s{seed}_up.png")
                    img_up.rotate(m["rot"], expand=True).save(out)
                rec = {"ok": True, "s": round(time.time() - t, 1), "attempt": attempt, "q": score_output(r, seed), "cm": False}
                try:
                    colormatch(r, seed)
                    rec["cm"] = True; rec["q_cm"] = score_output(r, seed, "_cm")
                except Exception as e:                         # noqa: BLE001
                    log(f"{uid}: retry colour match failed for seed {seed}: {e!r}")
                try:
                    fr = finalize(d, seed, "retry")
                    if fr:
                        rec["rgba"] = fr
                except Exception as e:                         # noqa: BLE001
                    log(f"{uid}: retry alpha finalize failed for seed {seed}: {e!r}")
                recs[str(seed)] = rec
                break
            log(f"{uid}: retry klein seed {seed} attempt {attempt} failed (rc {rc})")
            for _ in range(60):
                if gtt_mib() < 2500 and vram_mib() < 1000:
                    break
                time.sleep(2)
            recs[str(seed)] = {"ok": False, "attempt": attempt}
    ok = [k for k, v in recs.items() if v.get("ok")]
    m["retry"] = {"status": "done" if ok else "failed", "klein": recs}
    if ok:
        rb = max(ok, key=lambda k: _score(recs[k]))
        m["retry"]["best"] = rb
        ob = m.get("best")
        use_retry = ob is None or _score(recs[rb]) >= _score(m["klein"].get(str(ob))) - 0.15      # the text fix is the point: accept a slightly lower measured score
        m["final"] = {"src": "retry" if use_retry else "orig", "seed": rb if use_retry else ob}
    m["t_update"] = time.time()
    log(f"{uid}: retry {m['retry']['status']}, final {m.get('final')}")


def render_unit(m, cfg):
    uid = m["uid"]; d = UN / uid
    txt = (d / "prompt.txt").read_text()
    prompt = re.search(r"PROMPT:\s*(.+)", txt, re.S).group(1).strip()
    from PIL import Image
    W, H = Image.open(d / "in.png").size
    for seed in cfg["seeds"]:
        out = d / f"klein_s{seed}.png"
        for attempt in range(1, cfg["klein_tries"] + 1):
            t = time.time()
            rc = klein_run(d / "in.png", out, prompt, W, H, seed, d / f"klein_s{seed}_a{attempt}.log", d / f"klein_s{seed}_a{attempt}.err")
            if rc == 0 and out.exists() and out.stat().st_size > 1000:
                if m.get("lift"):                              # klein drew the brightened picture: keep it, darken the result back to the original tone
                    kl_b = Image.open(out); kl_b.save(d / f"klein_s{seed}_lifted.png")
                    apply_gamma(kl_b, 1.0 / m["lift"]).save(out)
                if m.get("rot"):                               # klein drew it upright: keep that copy, and rotate the result back to the original orientation
                    img_up = Image.open(out); img_up.save(d / f"klein_s{seed}_up.png")
                    img_up.rotate(m["rot"], expand=True).save(out)
                rec = {"ok": True, "s": round(time.time() - t, 1), "attempt": attempt, "q": score_output(d, seed), "cm": False}
                try:
                    colormatch(d, seed)
                    rec["cm"] = True; rec["q_cm"] = score_output(d, seed, "_cm")
                except Exception as e:                         # noqa: BLE001
                    log(f"{uid}: colour match failed for seed {seed}: {e!r}")
                try:                                           # final RGBA: alpha from the original where the silhouette agrees, green screen removed
                    fr = finalize(d, seed)
                    if fr:
                        rec["rgba"] = fr
                except Exception as e:                         # noqa: BLE001
                    log(f"{uid}: alpha finalize failed for seed {seed}: {e!r}")
                m["klein"][str(seed)] = rec
                break
            log(f"{uid}: klein seed {seed} attempt {attempt} failed (rc {rc}); GTT {gtt_mib()} MiB")
            for _ in range(60):
                if gtt_mib() < 2500 and vram_mib() < 1000:
                    break
                time.sleep(2)
            m["klein"][str(seed)] = {"ok": False, "attempt": attempt}
        m["t_update"] = time.time(); save_meta(m)
    ok = [s for s, v in m["klein"].items() if v.get("ok")]
    if ok:
        m["best"] = max(ok, key=lambda x: (m["klein"][x].get("q_cm") or m["klein"][x].get("q", {})).get("score", -9))
    m["status"] = "done" if ok else "failed"
    if not ok:
        m["error"] = "klein produced no image"
    m["t_update"] = time.time()


PV = 6          # prompt/analysis version; units written with an older one are archived and redone at start-up


def score_output(d, seed, suffix=""):
    """Deterministic checks of a klein result against the original: silhouette IoU (cut-outs), mean colour difference
    after downscaling, SSIM.  Never raises."""
    try:
        import numpy as np
        from PIL import Image
        from skimage import color as skc
        from skimage.metrics import structural_similarity
        orig = Image.open(d / "orig.png").convert("RGBA"); ow, oh = orig.size
        kl = Image.open(d / f"klein_s{seed}{suffix}.png").convert("RGB")
        a_o = np.asarray(orig.getchannel("A")).astype(np.float32) / 255
        cut = (a_o < 0.5).mean() > 0.02 or a_o.max() < 0.99 and (a_o < 0.98).mean() > 0.02
        arr = np.asarray(kl).astype(np.float32)
        k = arr[..., 1] - np.maximum(arr[..., 0], arr[..., 2])
        bg = ((k > 70) & (arr[..., 1] > 150)).astype(np.float32)           # green screen pixels
        fg = 1 - bg
        fg_s = np.asarray(Image.fromarray((fg * 255).astype(np.uint8)).resize((ow, oh), Image.BOX)).astype(np.float32) / 255
        res = {}
        mo = a_o >= 0.5 if cut else np.ones_like(a_o, bool)
        if cut:
            mk = fg_s >= 0.5
            inter, uni = (mo & mk).sum(), (mo | mk).sum()
            res["iou"] = round(float(inter / max(1, uni)), 3)
        small = np.asarray(kl.resize((ow, oh), Image.BOX)).astype(np.float32)
        ref = np.asarray(orig.convert("RGB")).astype(np.float32)
        both = mo & (fg_s >= 0.9) if cut else mo
        if both.sum() > 8:
            de = np.linalg.norm(skc.rgb2lab(small / 255) - skc.rgb2lab(ref / 255), axis=2)
            res["dE"] = round(float(de[both].mean()), 1)
            gl = lambda x: x @ np.array([0.299, 0.587, 0.114], np.float32)      # noqa: E731
            gm = float(gl(ref)[both].mean())
            a1, b1 = np.where(both, gl(ref), gm), np.where(both, gl(small), gm)
            res["ssim"] = round(float(structural_similarity(a1, b1, data_range=255, win_size=min(7, min(ow, oh) // 2 * 2 - 1 if min(ow, oh) < 14 else 7))), 3)
        flags = []
        if cut and res.get("iou", 1) < 0.8: flags.append("shape drift")
        if res.get("dE", 0) > 25: flags.append("colour drift")
        if res.get("ssim", 1) < 0.55: flags.append("structure drift")
        res["flags"] = flags
        res["score"] = round(res.get("ssim", 0) - res.get("dE", 0) / 100 + (res.get("iou", 1) if cut else 0), 3)
        return res
    except Exception as e:                                       # noqa: BLE001
        return {"error": repr(e)[:120], "flags": [], "score": -9}


_PREP_SEM = threading.Semaphore(20)                            # CPU-heavy part of preparing a unit (measurements, OCR): the machine has 20 threads, so this equals the cloud window


def prep_one(uid, batch_no, cfg, idx, tasks):
    """cloud mode, worker thread: prepare the unit and write its prompt.  Touches only the unit's own files (the main thread saves meta and index)."""
    if stop_flag["v"] or (PIPE / "STOP").exists():
        return None
    m = read_meta(uid)
    if m and m["status"] in ("prompted", "done"):
        return m
    with _PREP_SEM:
        m = m if (m and (UN / uid / "in.png").exists()) else new_meta(uid, idx, tasks, batch_no)
        m["batch"] = batch_no
        m["redo_token"] = (cfg.get("redo") or {}).get(uid, 0)
        if cfg.get("soft_route", True):
            try:
                if uid not in SOFT_SKIP and (m["stem"] in (cfg.get("soft_force") or []) or is_soft_effect(UN / uid / "orig.png")):
                    render_soft(m)
                    return m
            except Exception as e:                             # noqa: BLE001
                log(f"{uid}: soft-effect route failed ({e!r}); using the normal route")
    write_prompt(m, cfg)
    return m


def wait_gate():
    while (PIPE / "PAUSE").exists() and not stop_flag["v"]:
        push_state(phase="paused", detail="PAUSE file present: remove pipeline/PAUSE to continue")
        time.sleep(5)


def pending_list(uids, cfg):
    p = [u for u in uids if (INDEX.get(u) or {}).get("status") not in ("done", "failed")]
    try:                                                       # units redone because of review feedback or a redraw request go first
        urgent = {u for u, e in load_feedback().items() if not e.get("skipped")} | set(SOFT_SKIP)
        p = [u for u in p if u in urgent] + [u for u in p if u not in urgent]
    except Exception as e:                                     # noqa: BLE001
        log(f"priority ordering failed: {e!r}")
    return p


MAGENTA_REQ = PIPE / "magenta_requests.jsonl"
MAGENTA_SEEDS = [1, 2, 3, 4]


def magenta_prompt(m, cfg):
    """the unit's current prompt with the BACKGROUND changed from green to magenta: the cloud model rewrites it (only the screen colour; green subjects stay green), regex fallback"""
    d = UN / m["uid"]
    src = d / "retry" / "prompt.txt" if (m.get("final") or {}).get("src") == "retry" and (d / "retry" / "prompt.txt").exists() else d / "prompt.txt"
    prompt = re.search(r"PROMPT:\s*(.+)", src.read_text(), re.S).group(1).strip()
    c = cloud_cfg(cfg)
    if c:
        try:
            ask = ("Below is a prompt for an image-editing model. The picture is a cut-out on a flat green (#00FF00) chroma-key background. Rewrite the prompt so the background and every empty gap or hole is a flat, pure magenta (#FF00FF) screen instead, "
                   "with no shadow, glow or magenta spill on the subject. Change ONLY the screen: keep every green that is the colour of the subject itself (leaves, paint, lenses) exactly as green. Keep everything else word for word. Answer with the rewritten prompt only.\n\n" + prompt)
            body = {"model": c["model"], "stream": False, "max_tokens": 4000, "temperature": 0.2, "messages": [{"role": "user", "content": ask}]}
            req = urllib.request.Request(c["url"], json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + (ROOT / c.get("key_file", "gemini_api.key")).read_text().strip()})
            out = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"].strip()
            if len(out.split()) >= 40 and re.search(r"magenta|#FF00FF", out, re.I):
                return out
        except Exception as e:                                 # noqa: BLE001
            log(f"{m['uid']}: magenta prompt rewrite failed ({e!r}); using plain word replacement")
    p = prompt.replace("#00FF00", "#FF00FF")
    return re.sub(r"(?i)\b(pure |flat |solid |plain |chroma[- ]key )*green\b(?=\s+(?:chroma|screen|background|backdrop|surface|gap|field))", lambda mo: (mo.group(1) or "") + "magenta", p)


def render_magenta(m, cfg):
    """redraw on a flat MAGENTA screen (for green subjects and shadows on green) and key it with magentakey: results in units/<uid>/magenta/ as extra versions magenta:1..4"""
    import numpy as np, cv2
    from PIL import Image
    from magentakey import magenta_key
    uid = m["uid"]; d = UN / uid; r = d / "magenta"; r.mkdir(exist_ok=True)
    rec = {"status": "running", "t": time.time(), "klein": {}}
    m["magenta"] = rec; save_meta(m)
    if m.get("rot") or m.get("lift") or not m.get("cutout"):
        rec.update(status="failed", error="not supported for rotated, dark or opaque pictures"); return
    try:
        prompt = magenta_prompt(m, cfg)
        o = np.asarray(Image.open(d / "orig.png").convert("RGBA")).astype(np.float32) / 255
        vis = o[..., 3] > 0.3
        green_subj = bool(vis.any() and ((o[..., 1] > np.maximum(o[..., 0], o[..., 2]) + 0.012)[vis]).mean() > 0.3)      # green subject (trees, leaves, green paint): keep colours, despill the magenta fringe
        W, H = Image.open(d / "in.png").size
        pm = np.dstack([o[..., :3] * o[..., 3:4], o[..., 3:4]]); up = cv2.resize(pm, (W, H), interpolation=cv2.INTER_CUBIC); a = np.clip(up[..., 3:4], 0, 1)
        ref = np.clip(up[..., :3], 0, 1) + np.array([1, 0, 1], np.float32) * (1 - a)
        Image.fromarray((np.clip(ref, 0, 1) * 255 + .5).astype(np.uint8)).save(r / "in_magenta.png")
        (r / "prompt.txt").write_text("PROMPT: " + prompt)
        for seed in MAGENTA_SEEDS:
            out = r / f"klein_s{seed}.png"; t = time.time()
            if klein_run(r / "in_magenta.png", out, prompt, W, H, seed, r / f"klein_s{seed}.log", r / f"klein_s{seed}.err") != 0 or not out.exists():
                log(f"{uid}: magenta seed {seed} failed"); continue
            arr, G, k0 = magenta_key(np.asarray(Image.open(out).convert("RGB")).astype(np.float32), neutralize=not green_subj, green_subject=green_subj)
            Image.fromarray(arr, "RGBA").save(r / f"klein_s{seed}_rgba.png")
            rec["klein"][str(seed)] = {"ok": True, "s": round(time.time() - t, 1), "rgba": {"alpha": "magenta key"}}
            rec["t"] = time.time(); save_meta(m)
        rec["status"] = "done" if rec["klein"] else "failed"
        if not rec["klein"]:
            rec["error"] = "klein produced no image"
    except Exception as e:                                     # noqa: BLE001
        log(f"{uid}: magenta redraw failed: {e!r}")
        rec.update(status="failed", error=repr(e)[:200])
    rec["t"] = time.time()
    if rec["status"] == "done":                                # back into the swipe review, magenta versions first
        with open(PIPE / "reviews.jsonl", "a") as f:
            f.write(json.dumps({"uid": uid, "decision": "undo", "chosen": None, "seen": [], "issues": [], "text": "", "ts": time.time()}) + "\n")
    log(f"{uid}: magenta redraw {rec['status']} ({len(rec['klein'])} seeds)")


def magenta_pending():
    """uids with a request newer than their last magenta redraw"""
    try:
        reqs = [json.loads(l) for l in MAGENTA_REQ.read_text().splitlines() if l.strip()]
    except OSError:
        return []
    last = {}
    for q in reqs:
        last[q["uid"]] = max(last.get(q["uid"], 0), q.get("ts", 0))
    out = []
    for uid, ts in last.items():
        m = read_meta(uid)
        mg = (m or {}).get("magenta") or {}
        stale = mg.get("status") == "running" and time.time() - mg.get("t", 0) > 900       # an interrupted run (pipeline restarted) is redone
        if m and m.get("status") == "done" and (mg.get("t", 0) < ts or stale):
            out.append(uid)
    return out


def housekeeping(cfg, busy):
    """review requests, reject feedback and config 'redo' tokens (what the batch loop does once per batch); units in 'busy' are in flight and left alone"""
    try:
        process_redraw_requests()
        process_reject_feedback()
    except Exception as e:                                     # noqa: BLE001
        log(f"review requests failed: {e!r}")
    for ruid, tok in (cfg.get("redo") or {}).items():
        m0 = read_meta(ruid)
        if ruid not in busy and m0 and m0.get("redo_token", 0) < tok and m0.get("status") in ("done", "prompted", "failed"):
            dst = PIPE / "archive" / f"redo{tok}" / ruid
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(UN / ruid), str(dst))
            INDEX.pop(ruid, None)
            log(f"{ruid}: redo requested (token {tok}); old result archived to {dst}")
    push_index()


def cloud_rolling(uids, idx, tasks, already, batch_no):
    """cloud mode, rolling: `parallel` (20) prompts are always being written, finished prompts wait for klein without limit, so klein never waits for a prompt.
    The main thread owns klein, meta saving and the index; worker threads only write prompts (prep_one) and reviews (review_unit).
    Reviews of units redrawn in this run run in the background, and corrected prompts are redrawn (retry) between units.
    On STOP it finishes the current redraw, saves every prompt already written (those units stay 'prompted' and resume later) and returns."""
    from concurrent.futures import wait, FIRST_COMPLETED
    cfg = load_cfg(); c = cloud_cfg(cfg)
    window = int(c.get("parallel", 20))
    SERVER_ON["v"] = bool(c.get("klein_server", True)) and SERVER_ON["fails"] < 3
    wex = ThreadPoolExecutor(window); rex = ThreadPoolExecutor(int(c.get("review_parallel", 6)))
    wf, rf = {}, {}                                            # future -> uid / future -> meta
    magentas = []
    ready, retries = [], []                                    # metas whose prompt is written, waiting for klein / uids whose corrected prompt waits for klein
    sub, cool, rsub = set(), {}, set()
    n = 0; t_house = 0.0

    def take_writer(f, queue):
        uid = wf.pop(f)
        if f.cancelled():
            sub.discard(uid); return 0
        try:
            m = f.result()
        except Exception as e:                                 # noqa: BLE001
            log(f"{uid}: prompt preparation failed: {e!r}")
            STATE["last_error"] = f"{uid}: {e!r}"; sub.discard(uid); cool[uid] = time.time() + 120
            return 0
        if m is None:
            sub.discard(uid); return 0
        save_meta(m)
        if m.get("soft") and m["status"] == "done":
            STATE["run_done"] += 1; sub.discard(uid)
            log(f"{uid}: soft effect (shadow/smoke/glow): smooth enlargement, no redraw")
            return 1
        if m["status"] == "failed":
            STATE["last_error"] = f"{uid}: {m.get('error')}"; sub.discard(uid)
            return 1
        if m["status"] == "prompted" and queue:
            urgent_set = {u for u, e in load_feedback().items() if not e.get("skipped")}
            if m["uid"] in urgent_set:
                ready.insert(0, m)
                log(f"{m['uid']}: urgent reject prompt ready -> inserted at FRONT of image gen queue")
                prefetch_unit(m, cfg)
            else:
                ready.append(m)
            return 0
        sub.discard(uid)
        if m["status"] == "pending":
            cool[uid] = time.time() + 120                      # cloud trouble: not again for 2 minutes
        return 0

    def review_candidate(m):
        rcfg = cfg.get("review") or {}
        uid = m["uid"]
        if rcfg.get("enabled") and uid not in rsub and "review" not in m and m.get("status") == "done" and (uid in set(rcfg.get("force") or []) or needs_review(m)):
            rsub.add(uid)
            rf[rex.submit(review_unit, m, cfg)] = m

    for ruid in list(INDEX):                                   # leftovers of an earlier run: corrected prompts not yet redrawn, forced reviews
        if INDEX[ruid].get("status") == "done":
            mr = read_meta(ruid)
            if mr and (mr.get("retry") or {}).get("status") == "pending":
                retries.append(ruid)
            elif mr and ruid in set((cfg.get("review") or {}).get("force") or []):
                review_candidate(mr)
    try:
        while True:
            cfg = load_cfg(); c = cloud_cfg(cfg)
            stop = stop_flag["v"] or (PIPE / "STOP").exists() or not c
            if not stop and time.time() - t_house > 5.0:
                t_house = time.time()
                housekeeping(cfg, sub | set(retries))
                for mu in magenta_pending():
                    if mu not in magentas:
                        magentas.append(mu)
            pending = pending_list(uids, cfg)
            STATE["pending"] = len(pending)
            if not stop:
                urgent_set = {u for u, e in load_feedback().items() if not e.get("skipped")} | set(SOFT_SKIP)
                # Dispatch any real pending review reject re-prompts immediately
                urgent_pending = [u for u in pending if u in urgent_set and u not in sub and cool.get(u, 0) <= time.time()]
                for u in urgent_pending:
                    wf[wex.submit(prep_one, u, batch_no + n // window, cfg, idx, tasks)] = u
                    sub.add(u)
                    log(f"{u}: urgent reject priority -> Gemini prompt dispatched immediately")

                room = window - len(wf)                            # `window` Gemini requests in flight; finished prompts wait for klein without limit
                for u in pending:
                    if room <= 0:
                        break
                    if u in sub or cool.get(u, 0) > time.time():
                        continue
                    wf[wex.submit(prep_one, u, batch_no + n // window, cfg, idx, tasks)] = u
                    sub.add(u); room -= 1
            for f in [f for f in wf if f.done()]:
                n += take_writer(f, not stop)
            for f in [f for f in rf if f.done()]:
                mr = rf.pop(f)
                try:
                    f.result()
                except Exception as e:                         # noqa: BLE001
                    log(f"{mr['uid']}: review failed: {e!r}")
                save_meta(mr)
                if (mr.get("retry") or {}).get("status") == "pending":
                    retries.append(mr["uid"])
            if stop:
                break
            urgent_set = {u for u, e in load_feedback().items() if not e.get("skipped")} | set(SOFT_SKIP)
            urgent_ready = [m for m in ready if m["uid"] in urgent_set]
            if urgent_ready:
                m = urgent_ready[0]
                ready.remove(m)
                uid = m["uid"]
                prefetch_unit(m, cfg)
                wait_gate()
                push_state(phase="rendering", current=uid, unit_started=time.time(), detail=f"klein 9B is redrawing urgent reject {m['stem']}")
                render_unit(m, cfg)
                save_meta(m)
                sub.discard(uid); n += 1
                STATE["run_done"] += 1 if m["status"] == "done" else 0
                STATE["recent"] = (STATE["recent"] + [uid])[-12:]
                push_state(done_count=already + STATE["run_done"])
                log(f"{uid}: urgent reject redo {m['status']} (writer {m.get('writer', {}).get('s')}s, klein {[v.get('s') for v in m['klein'].values()]})")
                if m["status"] == "done":
                    review_candidate(m)
                continue
            if magentas:
                muid = magentas.pop(0); mm = read_meta(muid)
                if mm:
                    wait_gate()
                    push_state(phase="rendering", current=muid, unit_started=time.time(), detail=f"klein 9B is redrawing {mm['stem']} on a magenta screen")
                    render_magenta(mm, cfg)
                    save_meta(mm)
                continue
            if retries:
                ruid = retries.pop(0); mr = read_meta(ruid)
                if mr:
                    wait_gate()
                    push_state(phase="rendering", current=ruid, unit_started=time.time(), detail=f"klein 9B is redrawing {mr['stem']} with the corrected prompt")
                    render_retry(mr, cfg)
                    save_meta(mr)
                continue
            if ready:
                rank = {u: i for i, u in enumerate(pending)}        # the most urgent unit first (review rejects, redo requests, config priority), then the normal order
                m = min(ready, key=lambda x: rank.get(x["uid"], 10 ** 9)); ready.remove(m); uid = m["uid"]
                prefetch_unit(m, cfg)                              # queue this unit's jobs and the next unit's now: the server always has work waiting while this unit is post-processed
                if ready:
                    prefetch_unit(min(ready, key=lambda x: rank.get(x["uid"], 10 ** 9)), cfg)
                wait_gate()
                push_state(phase="rendering", current=uid, unit_started=time.time(), detail=f"klein 9B is redrawing {m['stem']} ({len(ready)} prompts ready, {len(wf)} being written)")
                render_unit(m, cfg)
                save_meta(m)
                sub.discard(uid); n += 1
                STATE["run_done"] += 1 if m["status"] == "done" else 0
                STATE["recent"] = (STATE["recent"] + [uid])[-12:]
                push_state(done_count=already + STATE["run_done"])
                log(f"{uid}: {m['status']} (writer {m.get('writer', {}).get('s')}s, klein {[v.get('s') for v in m['klein'].values()]})")
                if m["status"] == "done":
                    review_candidate(m)
                continue
            if not wf and not rf and not pending:
                break
            push_state(phase="writing", current=None, detail=f"waiting: {len(wf)} prompts being written, {len(rf)} reviews running")
            futs = list(wf) + list(rf)
            if futs:
                wait(futs, timeout=3, return_when=FIRST_COMPLETED)
            else:
                time.sleep(3)
    finally:
        wex.shutdown(wait=True, cancel_futures=True)           # writers in flight are short; their prompts are saved so the units resume as 'prompted'
        for f in list(wf):
            try:
                take_writer(f, False)
            except Exception as e:                             # noqa: BLE001
                log(f"drain after stop: {e!r}")
        rex.shutdown(wait=True)
        for f in list(rf):
            mr = rf.pop(f)
            try:
                f.result(); save_meta(mr)
            except Exception as e:                             # noqa: BLE001
                log(f"{mr['uid']}: review failed: {e!r}")
    return n


def main():
    PIPE.mkdir(parents=True, exist_ok=True); UN.mkdir(exist_ok=True)
    signal.signal(signal.SIGINT, lambda *a: stop_flag.update(v=True))
    signal.signal(signal.SIGTERM, lambda *a: stop_flag.update(v=True))
    cfg = load_cfg()
    load_soft_skip()
    uids, idx, tasks = universe(cfg)
    total = len(uids)
    arch = PIPE / "archive"
    for uid in uids:                                           # rebuild the index from earlier work (resume)
        m = read_meta(uid)
        if m and m.get("status") in ("done", "prompted") and m.get("pv", 0) < cfg.get("min_pv", PV):
            dst = arch / f"pv{m.get('pv', 0)}" / uid
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(UN / uid), str(dst))
            log(f"{uid}: archived (prompt version {m.get('pv', 0)} < {cfg.get('min_pv', PV)}) to {dst}")
            m = None
        if m and cfg.get("soft_route", True) and not m.get("soft") and m.get("status") in ("done", "prompted") and (UN / uid / "orig.png").exists():
            try:
                if uid not in SOFT_SKIP and (m.get("stem") in (cfg.get("soft_force") or []) or is_soft_effect(UN / uid / "orig.png")):
                    dst = arch / "softroute" / uid
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(UN / uid), str(dst))
                    log(f"{uid}: soft effect, klein result archived to {dst}; will be enlarged smoothly instead")
                    m = None
            except Exception as e:                             # noqa: BLE001
                log(f"{uid}: soft-effect check failed: {e!r}")
        if m:
            save_meta_quiet = m
            INDEX[uid] = {"uid": uid, "stem": m["stem"], "status": m["status"], "w": m["size"][0], "h": m["size"][1], "cutout": m["cutout"],
                          "t": m.get("t_update", 0), "batch": m.get("batch"), "seeds": sorted(int(s) for s in m.get("klein", {}) if m["klein"][s].get("ok")),
                          "writer_s": (m.get("writer") or {}).get("s"), "family": m.get("family"), "error": m.get("error")}
    push_index()
    already = sum(1 for r in INDEX.values() if r["status"] == "done")
    STATE.update(total=total, already_done=already, run_done=0)
    log(f"pipeline start: {total} units, {already} already done, config {cfg}")
    batch_no = max([r.get("batch") or 0 for r in INDEX.values()] + [0])
    processed = 0
    while not stop_flag["v"] and not (PIPE / "STOP").exists():
        cfg = load_cfg()
        STATE["writer"] = cloud_cfg(cfg)["model"] if cloud_cfg(cfg) else "Flash-Next"
        try:
            process_redraw_requests()
            process_reject_feedback()
        except Exception as e:                                 # noqa: BLE001
            log(f"review requests failed: {e!r}")
        for ruid, tok in (cfg.get("redo") or {}).items():           # config "redo": {uid: token}; a higher token archives the old result and redoes the unit
            m0 = read_meta(ruid)
            if m0 and m0.get("redo_token", 0) < tok and m0.get("status") in ("done", "prompted", "failed"):
                dst = PIPE / "archive" / f"redo{tok}" / ruid
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(UN / ruid), str(dst))
                INDEX.pop(ruid, None)
                log(f"{ruid}: redo requested (token {tok}); old result archived to {dst}")
        push_index()
        pending = [u for u in uids if (INDEX.get(u) or {}).get("status") != "done" and (INDEX.get(u) or {}).get("status") != "failed"]
        try:                                                   # units redone because of review feedback or a redraw request go first
            urgent = {u for u, e in load_feedback().items() if not e.get("skipped")} | set(SOFT_SKIP)
            pending = [u for u in pending if u in urgent] + [u for u in pending if u not in urgent]
        except Exception as e:                                 # noqa: BLE001
            log(f"priority ordering failed: {e!r}")
        if cfg.get("max_units") is not None:
            pending = pending[: max(0, cfg["max_units"] - processed)]
        STATE["pending"] = len(pending)
        if not pending:
            break
        if cloud_cfg(cfg):
            try:
                if strata_up():
                    stop_strata()                              # klein needs the whole GPU; the cloud model writes the prompts
                processed += cloud_rolling(uids, idx, tasks, already, batch_no)
            except Exception as e:                             # noqa: BLE001
                log(f"cloud run error: {e!r}")
                STATE["last_error"] = repr(e)
                time.sleep(30)
            batch_no = max([r.get("batch") or 0 for r in INDEX.values()] + [batch_no])
            continue
        batch = pending[: cfg["batch"]]
        batch_no += 1
        push_state(phase="writing", detail=f"batch {batch_no}: writing prompts", batch_no=batch_no, batch_units=batch,
                   done_count=already + STATE["run_done"], total=total)
        # ---- writer phase
        cloud = None
        try:
            if not cloud and any(not (UN / u / "prompt.txt").exists() or (read_meta(u) or {}).get("status") not in ("prompted", "done") for u in batch):
                start_strata()
            for uid in ([] if cloud else batch):
                if stop_flag["v"] or (PIPE / "STOP").exists():
                    break
                wait_gate()
                m = read_meta(uid)
                if m and m["status"] in ("prompted", "done"):
                    continue
                push_state(phase="writing", current=uid, unit_started=time.time(), detail=f"batch {batch_no}: prompt for {uid}")
                m = m if (m and (UN / uid / "in.png").exists()) else new_meta(uid, idx, tasks, batch_no)
                m["batch"] = batch_no
                m["redo_token"] = (cfg.get("redo") or {}).get(uid, 0)
                if cfg.get("soft_route", True):
                    try:
                        if uid not in SOFT_SKIP and (m["stem"] in (cfg.get("soft_force") or []) or is_soft_effect(UN / uid / "orig.png")):
                            render_soft(m); save_meta(m)
                            STATE["run_done"] += 1; processed += 1
                            log(f"{uid}: soft effect (shadow/smoke/glow): smooth enlargement, no redraw")
                            continue
                    except Exception as e:                     # noqa: BLE001
                        log(f"{uid}: soft-effect route failed ({e!r}); using the normal route")
                push_state(detail=f"batch {batch_no}: Flash-Next is writing the prompt for {m['stem']}")
                write_prompt(m, cfg)
                save_meta(m)
                if m["status"] == "failed":
                    STATE["last_error"] = f"{uid}: {m.get('error')}"
                    processed += 1
        except Exception as e:                                 # noqa: BLE001
            log(f"writer phase error: {e!r}")
            STATE["last_error"] = repr(e)
            time.sleep(10)
        # ---- review phase (Strata is up): Flash-Next checks finished redraws, once per unit, and queues a corrected retry if needed
        try:
            rcfg = cfg.get("review") or {}
            if rcfg.get("enabled") and not stop_flag["v"] and not (PIPE / "STOP").exists():
                force = set(rcfg.get("force") or [])
                cands = []
                for ruid in list(INDEX):
                    if INDEX[ruid].get("status") != "done":
                        continue
                    mr = read_meta(ruid)
                    if mr and "review" not in mr and mr.get("status") == "done" and (ruid in force or needs_review(mr)):
                        cands.append((0 if ruid in force else 1, mr.get("t_update", 0), ruid))
                cands = [c[2] for c in sorted(cands)][: int(rcfg.get("max_per_batch", 6))]
                if cands:
                    if not strata_up():
                        start_strata()
                    for ruid in cands:
                        if stop_flag["v"] or (PIPE / "STOP").exists():
                            break
                        wait_gate()
                        mr = read_meta(ruid)
                        push_state(phase="writing", current=ruid, unit_started=time.time(), detail=f"batch {batch_no}: Flash-Next is reviewing the redraw of {mr['stem']}")
                        review_unit(mr, cfg)
                        save_meta(mr)
        except Exception as e:                                 # noqa: BLE001
            log(f"review phase error: {e!r}")
        # ---- redraw phase
        todo = [u for u in batch if (read_meta(u) or {}).get("status") == "prompted"]
        if todo:
            if not cloud:
                stop_strata()
            for uid in todo:
                wait_gate()
                m = read_meta(uid)
                push_state(phase="rendering", current=uid, unit_started=time.time(), detail=f"batch {batch_no}: klein 9B is redrawing {m['stem']}")
                render_unit(m, cfg)
                save_meta(m)
                processed += 1
                STATE["run_done"] += 1 if m["status"] == "done" else 0
                STATE["recent"] = (STATE["recent"] + [uid])[-12:]
                push_state(done_count=already + STATE["run_done"], pending=max(0, STATE.get("pending", 0) - 1))
                log(f"{uid}: {m['status']} (writer {m.get('writer', {}).get('s')}s, klein {[v.get('s') for v in m['klein'].values()]})")
        else:
            processed += 0
        # ---- retries with the corrected prompts (Strata off, GPU free)
        try:
            retries = []
            for ruid in list(INDEX):
                if INDEX[ruid].get("status") == "done":
                    mr = read_meta(ruid)
                    if mr and (mr.get("retry") or {}).get("status") == "pending":
                        retries.append(ruid)
            if retries and not stop_flag["v"]:
                if strata_up():
                    stop_strata()
                for ruid in retries:
                    wait_gate()
                    mr = read_meta(ruid)
                    push_state(phase="rendering", current=ruid, unit_started=time.time(), detail=f"batch {batch_no}: klein 9B is redrawing {mr['stem']} with the corrected prompt")
                    render_retry(mr, cfg)
                    save_meta(mr)
        except Exception as e:                                 # noqa: BLE001
            log(f"retry phase error: {e!r}")
    push_state(phase="stopped" if (stop_flag["v"] or (PIPE / "STOP").exists()) else "finished", detail="run ended", current=None)
    try:
        stop_strata()
        kserver.stop()                                         # the loaded klein server holds ~9.5 GB VRAM; "GPU free" must be true when the run ends
    except Exception:                                          # noqa: BLE001
        pass
    push_state(phase="stopped" if (stop_flag["v"] or (PIPE / "STOP").exists()) else "finished", detail="run ended; GPU free", current=None)
    log("pipeline ended")


if __name__ == "__main__":
    main()
