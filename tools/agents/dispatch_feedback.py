#!/usr/bin/env python3
"""dispatch_feedback.py: sends review-page thumbs-down marked 'Send to agent' to the Gemini drawing agent that made the picture.

Reads pipeline/agent_inbox.jsonl (written by serve_gallery on a reject with route=agent). For each new entry it finds the workspace whose sprites.json holds the
unit (sprite_agent/ or button_agent/; an unknown unit is added to sprite_agent/ as group 'other'), waits until that workspace's agent session is idle, RESUMES the
session with the feedback (tools/agents/gemini_agent.sh -s), then queues the re-submitted pictures with queue_review.py so they re-enter the swipe review.
Progress: pipeline/agent_status.json (uid -> state), log tools/agents/dispatch.log. Stdlib only; run as the user service csvm-agent-dispatch."""
import json, os, subprocess, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / "pipeline"
INBOX, STATUS = PIPE / "agent_inbox.jsonl", PIPE / "agent_status.json"
PY = str(ROOT / "venv" / "bin" / "python")
LAUNCH = str(ROOT / "tools" / "agents" / "gemini_agent.sh")
WS = {"sprite_agent": ("sprites", ROOT / "sprite_agent"), "button_agent": ("buttons", ROOT / "button_agent")}
LOG = ROOT / "tools" / "agents" / "dispatch.log"


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def status():
    try:
        return json.loads(STATUS.read_text())
    except (OSError, ValueError):
        return {}


def set_status(uid, state, **kw):
    s = status(); s[uid] = dict(s.get(uid, {}), state=state, ts=time.time(), **kw)
    STATUS.write_text(json.dumps(s, indent=1))


def sid(session, workdir):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, session + str(Path(workdir).resolve())))


def busy(session, workdir):
    r = subprocess.run(["pgrep", "-f", "claude .*" + sid(session, workdir)], capture_output=True, text=True)
    return bool([p for p in r.stdout.split() if int(p) != os.getpid()])


def find(uid):
    for k, (session, d) in WS.items():
        try:
            sp = json.loads((d / "sprites.json").read_text())["sprites"]
        except (OSError, ValueError):
            continue
        for name, m in sp.items():
            if m["uid"] == uid:
                return k, name, False
    return None


def add_unknown(uid):
    d = WS["sprite_agent"][1]
    r = subprocess.run([PY, "-I", str(d / "add_sprites.py"), "other", uid], capture_output=True, text=True)
    log(f"added {uid} to sprite_agent: {r.stdout.strip()[-200:]} {r.stderr.strip()[-300:]}")
    return find(uid)


def prompt_for(items, newnames):
    lines = ["Feedback from the project owner from the swipe review. They saw your submitted picture and gave a thumbs-down with these notes. Re-read AGENT.md (the resume rules) and NOTES.md if needed,",
             "fix each sprite named below (change its recipe, re-run, `sprite_tool.py check`, look at the sheet, `submit`: a re-submit overwrites out/<name>.png), update STATUS.md, then answer in a few lines.", ""]
    for it in items:
        tags = ", ".join(it["issues"]) or "no tags"
        lines.append(f"- {it['name']}: problems tagged: {tags}." + (f" Their note: \"{it['text']}\"" if it["text"] else "") +
                     (" (NEW to you: this sprite was just added to your list, it has not been drawn by you yet; the picture they rejected was the older AI redraw in prev_redraw.png. Draw it as per AGENT.md, addressing the note.)" if it["name"] in newnames else ""))
    return "\n".join(lines)


def autoqueue():
    """every picture a drawing agent has submitted (out/*.png) becomes the unit's newest custom version, so it appears in the review (queue_review is idempotent)"""
    for k, (session, d) in WS.items():
        if (d / "out").is_dir() and any((d / "out").glob("*.png")):
            r = subprocess.run([PY, "-I", str(d / "queue_review.py")], capture_output=True, text=True)
            fresh = [l for l in r.stdout.splitlines() if "queued as" in l]
            if fresh:
                log(f"{k}: queued for review: {'; '.join(fresh)[:300]}")


def main():
    while True:
        autoqueue()
        try:
            entries = [json.loads(l) for l in INBOX.read_text().splitlines() if l.strip()] if INBOX.exists() else []
        except (OSError, ValueError):
            entries = []
        st = status()
        new = [e for e in entries if st.get(e["uid"], {}).get("handled_ts", 0) < e["ts"] and st.get(e["uid"], {}).get("state") != "working"]
        by_ws, newnames = {}, set()
        for e in new:
            f = find(e["uid"])
            if f is None:
                f = add_unknown(e["uid"])
                if f is None:
                    set_status(e["uid"], "failed", handled_ts=e["ts"], error="unit could not be added to an agent workspace")
                    continue
                newnames.add(f[1])
            by_ws.setdefault(f[0], {})[e["uid"]] = dict(e, name=f[1])       # newest feedback per unit wins
            set_status(e["uid"], "queued", name=f[1], workspace=f[0])
        for k, items in by_ws.items():
            session, d = WS[k]
            while busy(session, d):
                autoqueue()
                time.sleep(20)
            names = [i["name"] for i in items.values()]
            for uid in items:
                set_status(uid, "working", name=items[uid]["name"], workspace=k)
            pf = d / "feedback_prompt.txt"
            pf.write_text(prompt_for(list(items.values()), newnames))
            log(f"{k}: resuming {session} for {names}")
            r = subprocess.run([LAUNCH, "-d", str(d), "-s", session, "-t", "3600", "-f", str(pf)], stdin=subprocess.DEVNULL, capture_output=True, text=True)
            (d / "last_feedback_run.log").write_text(r.stdout[-6000:] + r.stderr[-1000:])
            q = subprocess.run([PY, "-I", str(d / "queue_review.py")] + names, capture_output=True, text=True)
            log(f"{k}: agent exit {r.returncode}; queue_review: {q.stdout.strip()[-300:]}")
            for uid, i in items.items():
                set_status(uid, "done" if r.returncode == 0 else "failed", name=i["name"], workspace=k, handled_ts=i["ts"])
        time.sleep(15)


if __name__ == "__main__":
    main()
