"""serve_gallery.py SITE_DIR HOST PORT TOKEN

Serves SITE_DIR read-only at http://HOST:PORT/TOKEN/ . Everything outside that prefix is a 404 and
directory listings are off, so the page is reachable only by someone who has the link. Bind HOST to the
machine's tailnet address so it is not exposed on the LAN. Follows the symlinks in site/full/ (they point
at the full-resolution results); resolves them only inside this tree's own out/ folder.
"""
import urllib.parse
import http.server, io, json, os, re, socketserver, subprocess, sys, threading, time, urllib.parse
from pathlib import Path

site, host, port, token = Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3]), sys.argv[4]
ROOT = Path(__file__).resolve().parents[1]
PIPE = (ROOT / "pipeline").resolve()
LIVE_HTML = (ROOT / "tools" / "pipeline" / "live.html").resolve()
REVIEW_HTML = (ROOT / "tools" / "pipeline" / "review.html").resolve()
REVIEWS = PIPE / "reviews.jsonl"                      # append-only log of the phone review decisions (this server also writes uploaded replacement pictures to units/<uid>/custom/)
LOCK = threading.Lock()
ALPHAFIX_LOCK = threading.Lock()
VERSION = re.compile(r"^(klein|retry|photo|custom|magenta):[0-9]{1,3}$")
ISSUES = {"lettering", "stray", "added", "missing", "shape", "colour", "material", "cartoon", "blurry", "edges", "tiling", "alpha", "other"}
EXP = (ROOT / "experiments").resolve()
allowed = [site, (ROOT / "out").resolve(), PIPE, LIVE_HTML, EXP]



def cloud_mode():
    """cloud-writer pipeline mode (config.json cloud_writer): results arrive fast, so the check-back lock is switched off"""
    try:
        c = json.loads((PIPE / "config.json").read_text()).get("cloud_writer")
        return bool(c) and c.get("enabled", True)
    except (OSError, ValueError, AttributeError):
        return False


def agent_held():
    """uids a drawing agent is working on right now: out of the review until redone. (a) feedback sent to an agent that the dispatcher has not finished
    (agent_status.json queued/working); (b) units on an agent's work list that are not SUBMITTED yet while that agent's session is running."""
    out = set()
    try:
        for uid, e in json.loads((PIPE / "agent_status.json").read_text()).items():
            if e.get("state") in ("queued", "working"):
                out.add(uid)
    except (OSError, ValueError, AttributeError):
        pass
    import uuid
    for wdir, session in (("sprite_agent", "sprites"), ("button_agent", "buttons")):
        d = ROOT / wdir
        try:
            sid = str(uuid.uuid5(uuid.NAMESPACE_URL, session + str(d.resolve())))
            if subprocess.run(["pgrep", "-f", "claude .*" + sid], capture_output=True).returncode != 0:
                continue                                       # the agent is not running: nothing in its list is being worked on
            work = json.loads((d / "sprites.json").read_text())["sprites"]
            st = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else {}
        except (OSError, ValueError, KeyError):
            continue
        for name, m in work.items():
            if st.get(name, {}).get("status") != "SUBMITTED":
                out.add(m["uid"])
    import hashlib                                             # (c) submitted but not yet added to the unit's custom versions (the dispatcher does that within seconds)
    for wdir in ("sprite_agent", "button_agent"):
        d = ROOT / wdir
        try:
            work = json.loads((d / "sprites.json").read_text())["sprites"]
            st = json.loads((d / "state.json").read_text())
        except (OSError, ValueError, KeyError):
            continue
        for name, e in st.items():
            if e.get("status") != "SUBMITTED" or name not in work:
                continue
            try:
                h = hashlib.sha1((d / "out" / f"{name}.png").read_bytes()).hexdigest()
            except OSError:
                continue
            try:
                items = json.loads((PIPE / "units" / work[name]["uid"] / "custom" / "custom.json").read_text()).get("items", [])
            except (OSError, ValueError):
                items = []
            if not any(it.get("sha1") == h for it in items):
                out.add(work[name]["uid"])
    return out


def held_units():
    """uids whose result is about to be replaced: a config redo token above the unit's own, or review feedback newer than the unit's last update"""
    out = set()
    try:
        if cloud_mode():
            raise ValueError("no check-back lock in cloud mode")
        # a committed check-back time is active: nothing new may be reviewed before it arrives, even if it is finished
        lk = json.loads(LOCKF.read_text())
        if lk.get("until", 0) > time.time() and json.loads((PIPE / "state.json").read_text()).get("pending", 1) != 0:
            return sorted(u["uid"] for u in json.loads((PIPE / "index.json").read_text()).get("units", []) if u.get("uid"))
    except (OSError, ValueError, AttributeError):
        pass
    try:
        cfg = json.loads((PIPE / "config.json").read_text())
    except (OSError, ValueError):
        cfg = {}
    def meta(uid):
        try:
            return json.loads((PIPE / "units" / uid / "meta.json").read_text())
        except (OSError, ValueError):
            return None
    for uid, tok in (cfg.get("redo") or {}).items():
        m = meta(uid)
        if m and m.get("redo_token", 0) < tok and m.get("status") in ("done", "prompted", "failed"):
            out.add(uid)
    try:
        fb = json.loads((PIPE / "feedback.json").read_text())
    except (OSError, ValueError):
        fb = {}
    for uid, e in fb.items():
        if e.get("skipped"):
            continue
        m = meta(uid)
        if m and m.get("status") == "done" and m.get("t_update", 0) <= e.get("ts", 0):
            out.add(uid)
    try:                                                       # a magenta-screen redo is waiting or running: out of the review until it is redone
        last = {}
        for ln in (PIPE / "magenta_requests.jsonl").read_text().splitlines():
            try:
                q = json.loads(ln); last[q["uid"]] = max(last.get(q["uid"], 0), q.get("ts", 0))
            except (ValueError, KeyError):
                continue
        for uid, ts in last.items():
            mg = (meta(uid) or {}).get("magenta") or {}
            if mg.get("t", 0) < ts or (mg.get("status") == "running" and time.time() - mg.get("t", 0) < 900):
                out.add(uid)
    except OSError:
        pass
    out |= agent_held()
    try:                                                       # units parked on purpose (pipeline/hidden.json), e.g. a superseded pass
        out |= set(json.loads((PIPE / "hidden.json").read_text()).get("uids", []))
    except (OSError, ValueError, AttributeError):
        pass
    return sorted(out)



def fresh_units():
    """uids whose latest decision is approve/original but that have a version made after it which the reviewer has not seen
    (a new klein/retry/magenta/photo redraw or a custom picture added later): they re-enter the review queue"""
    last = {}
    try:
        for ln in REVIEWS.read_text().splitlines():
            try:
                r = json.loads(ln); last[r["uid"]] = r
            except (ValueError, KeyError):
                continue
    except OSError:
        return []
    out = []
    for uid, r in last.items():
        if r.get("decision") not in ("approve", "original", "reject"):
            continue
        d = PIPE / "units" / uid
        if not d.is_dir():
            continue
        seen, ts, new = set(r.get("seen") or []), r.get("ts", 0), False
        for sub, kind in (("", "klein"), ("retry", "retry"), ("magenta", "magenta"), ("photo", "photo")):
            dd = d / sub if sub else d
            if dd.is_dir():
                for f in dd.glob("klein_s*.png"):
                    m = re.fullmatch(r"klein_s(\d+)\.png", f.name)
                    if m and f"{kind}:{m.group(1)}" not in seen and f.stat().st_mtime > ts:
                        new = True
        try:
            for it in json.loads((d / "custom" / "custom.json").read_text()).get("items", []):
                if it.get("id") not in seen and it.get("ts", 0) > ts:
                    new = True
        except (OSError, ValueError):
            pass
        if new:
            out.append(uid)
    return sorted(out)


MAX_UP = 40 * 1024 * 1024


def process_upload(uid, data, frm):
    """a picture uploaded from the review page, to be used instead of the redraws: normalised to PNG (long side <= 2048) under units/<uid>/custom/.
    Cut-out units need an alpha: the upload's own if it has one, else a green screen is keyed out, else the alpha of the version it replaces is used."""
    from PIL import Image
    import numpy as np
    Image.MAX_IMAGE_PIXELS = 64_000_000
    im = Image.open(io.BytesIO(data))
    if im.format not in ("PNG", "JPEG", "WEBP"):
        raise ValueError("only PNG, JPEG or WEBP pictures")
    im.load()
    udir = PIPE / "units" / uid
    meta = json.loads((udir / "meta.json").read_text())
    S_ = max(im.size)
    if S_ > 2048:
        im = im.resize((round(im.size[0] * 2048 / S_), round(im.size[1] * 2048 / S_)), Image.LANCZOS)
    rgba = im.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32)
    alpha_note, has_alpha = "opaque", False
    if meta.get("cutout") and not meta.get("soft"):
        a_up = arr[..., 3]
        if a_up.min() < 250:
            alpha_note, has_alpha = "its own transparency", True
        else:
            kk = arr[..., 1] - np.maximum(arr[..., 0], arr[..., 2])
            if (kk[:8, :8] > 110).mean() > 0.8 or (kk[-8:, -8:] > 110).mean() > 0.8:      # a green screen: soft key with colour unmixing (clouds, smoke, fine edges)
                sys.path.insert(0, str(ROOT / "tools" / "pipeline"))
                from greenkey import green_key
                arr = green_key(arr[..., :3])[0].astype(np.float32)
                alpha_note, has_alpha = "soft green-screen key with colour unmixing", True
            else:
                m = re.match(r"^(klein|retry):([0-9]{1,3})$", frm or "")
                if m:
                    f = udir / ("retry" if m.group(1) == "retry" else "") / f"klein_s{m.group(2)}_rgba.png"
                    if f.exists():
                        a = Image.open(f).convert("RGBA").getchannel("A").resize(rgba.size, Image.BICUBIC)
                        arr[..., 3] = np.asarray(a).astype(np.float32)
                        alpha_note, has_alpha = "transparency of the version it replaces", True
    out = Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8), "RGBA")
    if not has_alpha:
        out = out.convert("RGB")
    cdir = udir / "custom"
    cdir.mkdir(exist_ok=True)
    jp = cdir / "custom.json"
    try:
        items = json.loads(jp.read_text()).get("items", [])
    except (OSError, ValueError):
        items = []
    if len(items) >= 20:
        raise ValueError("too many uploads for this unit")
    n = max([int(it["id"].split(":")[1]) for it in items] + [0]) + 1
    fn = f"up{n}.png"
    out.save(cdir / fn)
    items.append({"id": f"custom:{n}", "file": fn, "rgba": has_alpha, "alpha": alpha_note, "size": list(out.size), "ts": time.time(), "replaces": frm})
    tmp = cdir / "custom.json.tmp"
    tmp.write_text(json.dumps({"items": items}))
    os.replace(tmp, jp)
    return items[-1]


def process_crop(uid, vid, crop_spec):
    from PIL import Image
    import numpy as np, cv2
    udir = PIPE / "units" / uid
    meta = json.loads((udir / "meta.json").read_text())
    src_file = None
    if vid.startswith("custom:"):
        cjson = json.loads((udir / "custom/custom.json").read_text())
        for it in cjson.get("items", []):
            if it.get("id") == vid:
                src_file = udir / "custom" / it["file"]
                break
    elif vid.startswith("klein:"):
        s = vid.split(":")[1]
        for candidate in [udir / f"klein_s{s}_rgba.png", udir / f"klein_s{s}_cm.png", udir / f"klein_s{s}.png"]:
            if candidate.exists():
                src_file = candidate; break
    elif vid.startswith("retry:"):
        s = vid.split(":")[1]
        for candidate in [udir / "retry" / f"klein_s{s}_rgba.png", udir / "retry" / f"klein_s{s}_cm.png", udir / "retry" / f"klein_s{s}.png"]:
            if candidate.exists():
                src_file = candidate; break
    elif vid.startswith("magenta:"):
        s = vid.split(":")[1]
        candidate = udir / "magenta" / f"klein_s{s}_rgba.png"
        if candidate.exists(): src_file = candidate

    if not src_file or not src_file.exists():
        raise ValueError(f"source file for {vid} not found")

    im = Image.open(src_file)
    w, h = im.size

    if crop_spec == "auto":
        arr = np.array(im.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)).mean(axis=0)
        gy = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)).mean(axis=1)
        thresh_x = np.percentile(gx, 85) * 0.75
        thresh_y = np.percentile(gy, 85) * 0.75
        peaks_x = np.where(gx > max(20.0, thresh_x))[0]
        peaks_y = np.where(gy > max(20.0, thresh_y))[0]
        x1, x2 = (int(peaks_x[0]), int(peaks_x[-1] + 1)) if len(peaks_x) >= 2 else (0, w)
        y1, y2 = (int(peaks_y[0]), int(peaks_y[-1] + 1)) if len(peaks_y) >= 2 else (0, h)
        box = (x1, y1, x2, y2)
    elif isinstance(crop_spec, dict):
        l = float(crop_spec.get("left", 0))
        t = float(crop_spec.get("top", 0))
        r = float(crop_spec.get("right", 0))
        b = float(crop_spec.get("bottom", 0))
        if l > 1 or t > 1 or r > 1 or b > 1:
            l /= 100.0; t /= 100.0; r /= 100.0; b /= 100.0
        x1 = int(round(l * w))
        y1 = int(round(t * h))
        x2 = int(round((1.0 - r) * w))
        y2 = int(round((1.0 - b) * h))
        box = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
    elif crop_spec == "target_ar":
        tw, th = meta.get("target") or [meta.get("w", 1), meta.get("h", 1)]
        target_ar = tw / th
        current_ar = w / h
        if current_ar > target_ar:
            nw = int(round(h * target_ar))
            x1 = (w - nw) // 2
            box = (x1, 0, x1 + nw, h)
        else:
            nh = int(round(w / target_ar))
            y1 = (h - nh) // 2
            box = (0, y1, w, y1 + nh)
    else:
        raise ValueError("unrecognized crop specification")

    if box[2] <= box[0] + 10 or box[3] <= box[1] + 10:
        raise ValueError("crop box too small")

    cropped = im.crop(box)
    target = meta.get("target")
    if target and len(target) == 2:
        out_im = cropped.resize((target[0], target[1]), Image.LANCZOS)
    else:
        out_im = cropped

    cdir = udir / "custom"
    cdir.mkdir(exist_ok=True)
    jp = cdir / "custom.json"
    try:
        items = json.loads(jp.read_text()).get("items", [])
    except Exception:
        items = []
    n = max([int(it["id"].split(":")[1]) for it in items if ":" in it.get("id", "")] + [0]) + 1
    fn = f"up{n}.png"
    out_im.save(cdir / fn)
    has_alpha = out_im.mode == "RGBA" and np.array(out_im.split()[-1]).min() < 250
    item = {
        "id": f"custom:{n}",
        "file": fn,
        "rgba": has_alpha,
        "alpha": "cropped image" if not has_alpha else "cropped image with alpha",
        "size": list(out_im.size),
        "ts": time.time(),
        "replaces": vid,
        "label": f"Cropped ({out_im.size[0]}×{out_im.size[1]})"
    }
    items.append(item)
    jp.write_text(json.dumps({"items": items}, indent=2))
    return item


LOCKF = PIPE / "review_lock.json"


def lock_call(b):
    """the 'check back at' time shown when the review queue is empty, kept on the server so every device shows the same committed time.
    {"action":"get"} reads it; "set" creates it unless an unexpired one exists (the existing one is returned, the time is never moved); "clear" removes it."""
    act = b.get("action")
    now = time.time()
    if act in ("get", "set", "clear") and cloud_mode():
        with LOCK:
            try:
                LOCKF.unlink()
            except OSError:
                pass
        return {"disabled": True}
    with LOCK:
        try:
            cur = json.loads(LOCKF.read_text())
        except (OSError, ValueError):
            cur = None
        if cur and not cur.get("until", 0) > now:
            cur = None
        if act == "clear":
            try:
                LOCKF.unlink()
            except OSError:
                pass
            return None
        if act == "set" and cur is None:
            hours = float(b.get("hours", 5))
            if not 0.1 <= hours <= 24:
                raise ValueError("bad hours")
            cur = {"set": now, "until": now + hours * 3600, "hours": hours, "n": max(0, min(int(b.get("n", 0)), 100000)),
                   "perHour": max(0.0, min(float(b.get("perHour", 0)), 10000.0)), "left": (None if b.get("left") is None else max(0, int(b["left"])))}
            tmp = LOCKF.with_suffix(".tmp")
            tmp.write_text(json.dumps(cur))
            os.replace(tmp, LOCKF)
        elif act not in ("get", "set", "clear"):
            raise ValueError("bad action")
        return cur


class H(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        path = urllib.parse.unquote(path.split("?", 1)[0].split("#", 1)[0])
        pre = f"/{token}/"
        if path == f"/{token}":
            path = pre
        if not path.startswith(pre):
            return str(site / "__nope__")
        rel = path[len(pre):]
        if rel.startswith("live/"):                       # the live pipeline page: its html, two json files, and the per-unit images
            sub = rel[5:]
            if sub == "":
                return str(LIVE_HTML)
            if sub == "review/":
                return str(REVIEW_HTML)
            if sub in ("state.json", "index.json"):
                return str(PIPE / sub)
            if sub.startswith("u/"):
                p = (PIPE / "units" / sub[2:]).resolve()
                return str(p) if (PIPE / "units").resolve() in p.parents else str(site / "__nope__")
            return str(site / "__nope__")
        if rel.startswith("exp/"):                        # experiment result pages and images (read-only)
            p = (EXP / rel[4:]).resolve()
            if p.is_dir():
                p = p / "index.html"
            return str(p) if EXP in p.parents else str(site / "__nope__")
        p = (site / rel.lstrip("/")).resolve() if rel else site / "index.html"
        if not any(a == p or a in p.parents for a in allowed):
            return str(site / "__nope__")
        return str(p)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in (f"/{token}/live", f"/{token}/live/review"):     # relative links need the trailing slash
            self.send_response(301); self.send_header("Location", path + "/"); self.end_headers(); return
        if path == f"/{token}/live/api/decisions":                # latest decision per unit: {uid: {decision, ts, chosen}}
            out = {}
            with LOCK:
                try:
                    lines = REVIEWS.read_text().splitlines() if REVIEWS.exists() else []
                except OSError:
                    lines = []
            for ln in lines:
                try:
                    r = json.loads(ln)
                    out[r["uid"]] = {"decision": r["decision"], "ts": r["ts"], "chosen": r.get("chosen")}
                except (ValueError, KeyError):
                    continue
            return self._json(200, out)
        if path == f"/{token}/live/api/fresh":                    # approved/original units with a newer unseen version: back in the review queue
            return self._json(200, fresh_units())
        if path == f"/{token}/live/api/held":                     # units waiting for a redo: they must not show up in the swipe review
            return self._json(200, held_units())
        super().do_GET()

    def do_POST(self):
        pth = self.path.split("?", 1)[0]
        if pth == f"/{token}/live/api/upload" and self.headers.get("X-Requested-With") == "review":
            try:
                q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                uid = (q.get("uid") or [""])[0]; frm = (q.get("from") or [""])[0]
                n = int(self.headers.get("Content-Length", "0"))
                if not (re.fullmatch(r"[A-Za-z0-9_&'-][A-Za-z0-9_.&' -]{0,79}", uid) and (PIPE / "units" / uid / "meta.json").is_file()):
                    return self._json(400, {"error": "bad unit"})
                if not 0 < n <= MAX_UP:
                    return self._json(413, {"error": "picture too large (max 40 MB)"})
                data = self.rfile.read(n)
                with LOCK:
                    item = process_upload(uid, data, frm)
                return self._json(200, {"ok": True, "item": item})
            except Exception as e:                                 # noqa: BLE001
                return self._json(400, {"error": str(e)[:200] or "bad picture"})
        if pth == f"/{token}/live/api/lock" and self.headers.get("X-Requested-With") == "review":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 2000:
                    return self._json(413, {"error": "bad size"})
                b = json.loads(self.rfile.read(n))
                return self._json(200, {"lock": lock_call(b)})
            except (ValueError, OSError, TypeError):
                return self._json(400, {"error": "bad request"})
        if pth == f"/{token}/live/api/magenta" and self.headers.get("X-Requested-With") == "review":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 1000:
                    return self._json(413, {"error": "bad size"})
                uid = json.loads(self.rfile.read(n)).get("uid")
                mp = PIPE / "units" / str(uid) / "meta.json"
                if not (isinstance(uid, str) and re.fullmatch(r"[A-Za-z0-9_&'-][A-Za-z0-9_.&' -]{0,79}", uid) and mp.is_file()):
                    return self._json(400, {"error": "bad unit"})
                m = json.loads(mp.read_text())
                if not m.get("cutout") or m.get("soft") or m.get("rot") or m.get("lift"):
                    return self._json(422, {"error": "this one can't be redrawn on magenta (not a plain cut-out)"})
                with ALPHAFIX_LOCK, open(PIPE / "magenta_requests.jsonl", "a") as f:
                    f.write(json.dumps({"uid": uid, "ts": time.time()}) + "\n")
                return self._json(200, {"ok": True})
            except (ValueError, OSError, TypeError):
                return self._json(400, {"error": "bad request"})
        if pth == f"/{token}/live/api/gemini_redo" and self.headers.get("X-Requested-With") == "review":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 1000:
                    return self._json(413, {"error": "bad size"})
                uid = json.loads(self.rfile.read(n)).get("uid")
                mp = PIPE / "units" / str(uid) / "meta.json"
                if not (isinstance(uid, str) and re.fullmatch(r"[A-Za-z0-9_&'-][A-Za-z0-9_.&' -]{0,79}", uid) and mp.is_file()):
                    return self._json(400, {"error": "bad unit"})
                rec = {
                    "uid": uid,
                    "decision": "reject",
                    "chosen": None,
                    "seen": [],
                    "issues": ["gemini_redo"],
                    "text": "Redo with Gemini prompt requested from review UI",
                    "ts": time.time()
                }
                with LOCK, open(REVIEWS, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                return self._json(200, {"ok": True})
            except (ValueError, OSError, TypeError):
                return self._json(400, {"error": "bad request"})
        if pth == f"/{token}/live/api/crop" and self.headers.get("X-Requested-With") == "review":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 4000:
                    return self._json(413, {"error": "bad size"})
                b = json.loads(self.rfile.read(n))
                uid = b.get("uid")
                vid = b.get("version")
                crop_spec = b.get("crop")
                mp = PIPE / "units" / str(uid) / "meta.json"
                if not (isinstance(uid, str) and re.fullmatch(r"[A-Za-z0-9_&'-][A-Za-z0-9_.&' -]{0,79}", uid) and mp.is_file()):
                    return self._json(400, {"error": "bad unit"})
                with LOCK:
                    item = process_crop(uid, vid, crop_spec)
                return self._json(200, {"ok": True, "item": item})
            except Exception as e:
                return self._json(400, {"error": str(e)[:200] or "bad crop"})
        if pth == f"/{token}/live/api/alphafix" and self.headers.get("X-Requested-With") == "review":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 4000:
                    return self._json(413, {"error": "bad size"})
                b = json.loads(self.rfile.read(n))
                uid, act = b.get("uid"), b.get("action")
                if not (isinstance(uid, str) and re.fullmatch(r"[A-Za-z0-9_&'-][A-Za-z0-9_.&' -]{0,79}", uid) and (PIPE / "units" / uid / "meta.json").is_file()):
                    return self._json(400, {"error": "bad unit"})
                py = str(ROOT / "venv" / "bin" / "python")
                kind = b.get("kind", "alpha")
                if kind not in ("alpha", "dirt"):
                    return self._json(400, {"error": "bad kind"})
                tool = str(ROOT / "tools" / "pipeline" / ("dirt_layer.py" if kind == "dirt" else "alpha_recipes.py"))
                if act == "options":
                    vid = b.get("version")
                    if not (isinstance(vid, str) and re.fullmatch(r"(klein|retry):[0-9]{1,3}", vid)):
                        return self._json(400, {"error": "bad version"})
                    cmd = [py, "-I", tool, "options", uid, vid]
                elif act == "apply":
                    opt = b.get("option")
                    if not (isinstance(opt, str) and re.fullmatch(r"[a-z_]{1,30}", opt)):
                        return self._json(400, {"error": "bad option"})
                    cmd = [py, "-I", tool, "apply", uid, opt]
                    if kind == "dirt":
                        vid = b.get("version")
                        if not (isinstance(vid, str) and re.fullmatch(r"(klein|retry):[0-9]{1,3}", vid)):
                            return self._json(400, {"error": "bad version"})
                        cmd.append(vid)
                elif act == "none":
                    text = str(b.get("text") or "")[:1000]; vid = str(b.get("version") or "")[:20]
                    with ALPHAFIX_LOCK, open(PIPE / "alphafix_requests.jsonl", "a") as f:
                        f.write(json.dumps({"uid": uid, "version": vid, "text": text, "ts": time.time()}) + "\n")
                    return self._json(200, {"ok": True})
                else:
                    return self._json(400, {"error": "bad action"})
                with ALPHAFIX_LOCK:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                out = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {"error": "no answer"}
                if isinstance(out, dict) and out.get("error"):
                    return self._json(422, out)
                return self._json(200, {"ok": True, "result": out})
            except (ValueError, OSError, TypeError, subprocess.TimeoutExpired, IndexError):
                return self._json(400, {"error": "bad request"})
        if pth != f"/{token}/live/api/review" or self.headers.get("X-Requested-With") != "review":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if not 0 < n <= 20000:
                return self._json(413, {"error": "bad size"})
            b = json.loads(self.rfile.read(n))
            uid, dec = b.get("uid"), b.get("decision")
            if not (isinstance(uid, str) and re.fullmatch(r"[A-Za-z0-9_&'-][A-Za-z0-9_.&' -]{0,79}", uid) and (PIPE / "units" / uid).is_dir()) or dec not in ("approve", "reject", "undo", "unsure", "original", "photo", "redraw"):
                return self._json(400, {"error": "bad request"})
            chosen = b.get("chosen")
            if chosen is not None and not (isinstance(chosen, str) and VERSION.match(chosen)):
                return self._json(400, {"error": "bad version"})
            seen = [x for x in (b.get("seen") or [])[:10] if isinstance(x, str) and VERSION.match(x)]
            issues = [x for x in (b.get("issues") or [])[:12] if x in ISSUES]
            text = b.get("text") or ""
            if not isinstance(text, str):
                return self._json(400, {"error": "bad text"})
            rec = {"uid": uid, "decision": dec, "chosen": chosen, "seen": seen, "issues": issues, "text": text[:1500], "ts": time.time()}
            if dec == "reject" and b.get("route") == "agent":
                rec["route"] = "agent"                         # redone by a drawing agent (tools/agents/dispatch_feedback.py), not by the FLUX pipeline
            with LOCK:
                with open(REVIEWS, "a") as f:
                    f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
                if rec.get("route") == "agent":
                    with open(PIPE / "agent_inbox.jsonl", "a") as f:
                        f.write(json.dumps({"uid": uid, "version": chosen, "issues": issues, "text": rec["text"], "ts": rec["ts"]}) + "\n")
            return self._json(200, {"ok": True, "ts": rec["ts"]})
        except (ValueError, OSError):
            return self._json(400, {"error": "bad request"})

    def list_directory(self, path):
        self.send_error(404)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, *a):
        pass


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


os.chdir(site)
S((host, port), H).serve_forever()
