"""detect.py: faces in every UI unit's original and chosen picture -> face_audit/faces.json + embeddings.npz.
YuNet finds faces and 5 landmarks, SFace embeds the aligned face. Originals are upscaled 4x first (their faces are tiny)."""
import json, sys, warnings
from pathlib import Path
import cv2, numpy as np
from PIL import Image
warnings.filterwarnings("ignore")
ROOT = Path("/home/deck/workspace/csvm-hd"); sys.path.insert(0, str(ROOT / "tools"))
import export_winners as ew
M = ROOT / "face_audit/models"
det = cv2.FaceDetectorYN.create(str(M / "yunet.onnx"), "", (320, 320), 0.6, 0.3, 5000)
rec = cv2.FaceRecognizerSF.create(str(M / "sface.onnx"), "")

def faces(img_rgb, up=1.0):
    im = cv2.cvtColor(np.asarray(img_rgb), cv2.COLOR_RGB2BGR)
    if up != 1.0:
        im = cv2.resize(im, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    h, w = im.shape[:2]
    # big pictures: detect on <=1600 px
    sc = min(1.0, 1600 / max(h, w)); d = cv2.resize(im, None, fx=sc, fy=sc) if sc < 1 else im
    det.setInputSize((d.shape[1], d.shape[0])); _, f = det.detect(d)
    out = []
    if f is None: return out
    for r in f:
        r = r.copy(); r[:14] /= sc
        r[:14] /= up
        # embed from the full-resolution (upscaled) picture
        rr = r.copy(); rr[:14] *= up
        al = rec.alignCrop(im, rr); e = rec.feature(al).flatten()
        out.append({"box": [float(v) for v in r[:4]], "lm": [float(v) for v in r[4:14]], "score": float(r[14]), "emb": e})
    return out

def main():
    last = {}
    for l in open(ROOT / "pipeline/reviews.jsonl"):
        if l.strip():
            r = json.loads(l); last[r["uid"]] = r
    rows, embs = [], []
    uids = sorted(u for u in last if u.startswith(("rimage_", "scrap_", "gfx_")))
    for k, uid in enumerate(uids):
        r = last[uid]
        og = ROOT / "pipeline/units" / uid / "orig.png"
        if not og.exists(): continue
        try:
            fo = faces(Image.open(og).convert("RGB"), 4.0 if max(Image.open(og).size) < 700 else 1.0)
            fc = []
            if r["decision"] == "approve" and r.get("chosen"):
                cf = ew.get_chosen_file(uid, r["chosen"])
                if cf and Path(cf).exists():
                    fc = faces(Image.open(cf).convert("RGB"))
        except Exception as e:
            print("ERR", uid, e); continue
        for kind, fl in (("orig", fo), ("chosen", fc)):
            for i, f in enumerate(fl):
                embs.append(f.pop("emb")); f.update(uid=uid, kind=kind, idx=i, emb_row=len(embs) - 1); rows.append(f)
        if k % 100 == 0: print(k, len(uids), len(rows), flush=True)
    json.dump(rows, open(ROOT / "face_audit/faces.json", "w"))
    np.save(ROOT / "face_audit/embs.npy", np.array(embs, np.float32))
    print("faces", len(rows), "units", len(uids))
main()
