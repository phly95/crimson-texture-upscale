"""blank_button.py: make ONE blank high-res master button plate with FLUX klein, as a normal review unit (ui_blank_button_master).

The 'brass' state (2nd of 4) of a source button sheet is cut out of the original, the lettering is asked away in the prompt, and four seeds are drawn,
colour matched and keyed like every other unit. The unit then appears in the swipe review; once a seed is approved, the drawing agent uses it as the base plate
for all lettered buttons (stretching it to each size and recolouring it per state) and only adds the lettering on top.
usage: venv/bin/python tools/pipeline/blank_button.py [SOURCE_UID] [SEED ...]"""
import json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run                                                                       # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "gfx_PF_B_AUDIO__2ded3062"
SEEDS = [int(x) for x in sys.argv[2:]] or [1, 2, 3, 4]
UID = "ui_blank_button_master__bb000001"
d = run.UN / UID
d.mkdir(parents=True, exist_ok=True)
src = Image.open(run.UN / SRC / "orig.png").convert("RGBA")
sh = src.height // 4
state = src.crop((0, sh, src.width, 2 * sh))                                      # the brass state: plate colour of states 2-4
state.save(d / "orig.png")
W, H = state.width * 5, state.height * 5                                          # 1105 x 210: the 5x target of the UI art
cw, ch = int(round(W / 16) * 16), int(round(H / 16) * 16)                         # klein wants multiples of 16
green = Image.new("RGBA", state.size, (0, 255, 0, 255))
green.alpha_composite(state)
green.convert("RGB").resize((cw, ch), Image.LANCZOS).save(d / "in.png")
prompt = ("Redraw this user interface button as a crisp, high-resolution picture on a flat pure green #00FF00 background. It is a horizontal capsule button with rounded ends, "
          "a thick dark gunmetal tubular steel frame, small square rivet brackets at both ends and a thin metal support bar under it. The panel is aged brass-tan parchment with a soft "
          "mottled texture and faint decorative filigree. IMPORTANT: the button must be completely BLANK. No letters, no words, no text, no numbers, no symbols and no marks on the panel: "
          "remove the lettering from the input and fill that area with the same plain mottled parchment as the rest of the panel. Keep the exact outline, proportions, "
          "frame and layout of the input. Sharp clean metal bevels with realistic highlights, no cartoon outlines, no green tint or green spill on the button.")
(d / "prompt.txt").write_text("PROMPT: " + prompt)
(d / "hint.txt").write_text("blank master plate for the lettered menu buttons; source " + SRC)
if not run.kserver.up():
    run.kserver.start(run.PIPE / "klein_server.log")
meta = {"uid": UID, "stem": "ui_blank_button_master", "task": "blank_button", "family": "ui", "size": [state.width, state.height], "target": [W, H], "cutout": True,
        "alpha_class": "Full", "tileable": False, "look": "flat-graphic-likely", "chapters": [], "klein": {}, "status": "running", "pv": run.PV, "batch": 0,
        "writer": {"s": 0.0, "tokens": 0, "tries": 1, "model": "hand-written prompt (blank master plate)"}, "t_update": time.time(), "ui_note": "blank master plate"}
for seed in SEEDS:
    t0 = time.time()
    out = d / f"klein_s{seed}.png"
    run.kserver.gen(d / "in.png", out, prompt, cw, ch, seed, timeout=600)
    rec = {"ok": True, "s": round(time.time() - t0, 1), "attempt": 1, "q": run.score_output(d, seed), "cm": False}
    try:
        run.colormatch(d, seed); rec["cm"] = True; rec["q_cm"] = run.score_output(d, seed, "_cm")
    except Exception as e:                                                       # noqa: BLE001
        print("colormatch failed", repr(e))
    try:
        fr = run.finalize(d, seed)
        if fr:
            rec["rgba"] = fr
    except Exception as e:                                                       # noqa: BLE001
        print("finalize failed", repr(e))
    meta["klein"][str(seed)] = rec
    print(f"seed {seed}: {rec['s']} s")
meta["best"] = max(meta["klein"], key=lambda x: (meta["klein"][x].get("q_cm") or meta["klein"][x].get("q", {})).get("score", -9))
meta["status"] = "done"; meta["t_update"] = time.time()
(d / "meta.json").write_text(json.dumps(meta))                                    # NOT run.save_meta: it rewrites index.json from the (empty) in-memory INDEX and wipes the review index
idx = json.loads((run.PIPE / "index.json").read_text())
idx["units"] = [u for u in idx["units"] if u["uid"] != UID]
idx["units"].insert(0, {"uid": UID, "stem": meta["stem"], "status": "done", "w": state.width, "h": state.height, "cutout": True, "t": time.time(), "batch": 0,
                        "seeds": sorted(int(s) for s in meta["klein"]), "writer_s": 0.0, "family": "ui", "error": None})
(run.PIPE / "index.json").write_text(json.dumps(idx))
run.kserver.stop()
print("done:", d)
