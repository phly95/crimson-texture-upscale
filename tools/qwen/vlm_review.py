"""vlm_review.py UNIT_DIR SEED -> prints REVIEW / VERDICT / PROMPT.
Follow-up for a finished redraw: Flash-Next (Strata on :9931) sees the ORIGINAL picture it described and klein's REDRAW of it, finds what went wrong
(misspelled or stray lettering, numbers painted on the picture, wrong shape/structure/colour) and, if needed, writes a corrected prompt for a second try."""
import base64, json, os, re, sys, time, urllib.request
from pathlib import Path

d = Path(sys.argv[1]); seed = sys.argv[2]
meta = json.loads((d / "meta.json").read_text())
prev = (d / "prompt.txt").read_text().strip()
digest = (d / "digest.txt").read_text().strip() if (d / "digest.txt").exists() else ""
hint = meta.get("hint", "")
red = next((p for p in (d / f"klein_s{seed}_lifted.png", d / f"klein_s{seed}_up.png", d / f"klein_s{seed}.png") if p.exists()), None)
q = meta["klein"].get(str(seed), {}).get("q_cm") or meta["klein"].get(str(seed), {}).get("q") or {}
auto = []
try:
    sys.path.insert(0, "/home/deck/workspace/csvm-hd/tools/pipeline")
    import textcheck
    pr = re.search(r"PROMPT:\s*(.+)", prev, re.S).group(1)
    exp = textcheck.expected_words(pr)
    if exp:
        auto.append("The prompt asked for these words: " + ", ".join(exp) + ". Check every letter of the lettering in the redraw against them.")
except Exception:                                                  # noqa: BLE001
    pass
if q.get("flags"):
    auto.append("Automatic measurements flagged: " + ", ".join(q["flags"]) + f" (shape overlap {q.get('iou')}, colour difference {q.get('dE')}, structure similarity {q.get('ssim')}).")


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


text = f"""You are the same prompt writer as before. FIRST picture: the tiny original texture from the video game Crimson Skies, enlarged ({hint}). SECOND picture: what the image editor (FLUX.2 klein) drew from your prompt (a green background is a chroma-key screen).

What you wrote before:
{prev}

Measurements of the original that you were given:
{digest[:3500]}

{' '.join(auto)}

Look at the redraw carefully, compare it with the original and with your prompt. Look especially for: misspelled, doubled, missing or extra letters in any lettering (compare letter by letter), including hyphens, dots or gaps painted inside words; stray digits, letters or symbols painted on the picture (any numbers you wrote in the prompt, such as coordinates, percentages or RGB values, can be painted as visible text); wrong subject, pose, layout or shape; missing or invented objects; wrong colours or materials.

Reply in exactly this format:
REVIEW: 2-5 short sentences saying concretely what went wrong, or that nothing is wrong.
VERDICT: OK or REDO (REDO only if a problem is clearly visible and a better prompt can fix it)
PROMPT: only if REDO: a complete new prompt of 110-170 words for the editor, same rules as before (begins with an imperative verb, keeps everything that was right, fixes what went wrong). Fix text problems by quoting only the exact words once, each with its letter count, never writing hyphens, dots or spaces between their letters (the editor paints them as marks), and add that there is no other text, number or symbol anywhere; describe positions in words (left, upper third) instead of numbers, and do not write coordinates or percentages at all."""

content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(d / "in.png")}}]
if red:
    content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(red)}})
content.append({"type": "text", "text": text})
body = {"model": os.environ.get("VLM_MODEL", "strata"), "stream": False, "max_tokens": 16000, "temperature": 0.6, **json.loads(os.environ.get("VLM_EXTRA", '{"reasoning_effort": "high"}')),
        "messages": [{"role": "user", "content": content}]}
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(os.environ.get("VLM_URL", "http://127.0.0.1:9931/v1/chat/completions"), json.dumps(body).encode(), {"Content-Type": "application/json", **({"Authorization": "Bearer " + open(os.environ["VLM_KEY_FILE"]).read().strip()} if os.environ.get("VLM_KEY_FILE") else {})}), timeout=1200))
print(r["choices"][0]["message"]["content"].strip())
print(f"[{time.time()-t:.0f}s, usage {r.get('usage')}]", file=sys.stderr)
