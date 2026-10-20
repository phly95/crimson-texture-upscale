"""headtex_score.py: pass/fail criteria for a redrawn head view (see face_audit/headtex_criteria.md). YuNet landmarks (eyes, nose tip, mouth corners) of the REFERENCE render are the layout ground truth.
score_view(ref_rgb, cand_rgb) -> dict of metrics + 'fail' list"""
import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]; M = ROOT / "face_audit/models"
_det = cv2.FaceDetectorYN.create(str(M / "yunet.onnx"), "", (1024, 1024), 0.3, 0.3, 5000); _rec = cv2.FaceRecognizerSF.create(str(M / "sface.onnx"), "")
THR = dict(lowcorr=0.97, lm_err=0.07, nose_axis=0.04, mouth_tilt=0.04, id_cos=0.5, mouth_corr=0.90, gog=0.90, strap=0.85, skin=0.80, nose_dx=4, nose_sym=0.88, mouth_err=0.07)
def face(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR); _det.setInputSize((bgr.shape[1], bgr.shape[0])); _, r = _det.detect(bgr)
    if r is None: return None
    r = r[np.argmax(r[:, 14])]; return dict(box=r[:4].copy(), lm=r[4:14].reshape(5, 2).copy(), score=float(r[14]), row=r, bgr=bgr)
def embed(f): return _rec.feature(_rec.alignCrop(f["bgr"], f["row"])).flatten()
def lowcorr(a, b, s=6):
    g = lambda x: cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (0, 0), s).astype(np.float32)
    a, b = g(a), g(b); a -= a.mean(); b -= b.mean(); return float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum()))
import json
GT = json.load(open(ROOT / "face_audit/headtex_gt.json"))
def score_view(ref, cand, frontal=False, gt=None):
    m = dict(lowcorr=lowcorr(ref, cand)); fail = []
    if m["lowcorr"] < THR["lowcorr"]: fail.append("layout drift (lowcorr %.3f < %.2f)" % (m["lowcorr"], THR["lowcorr"]))
    fr, fc = face(ref), face(cand)
    if fc is None: m["face"] = False; fail.append("no face found"); return m, fail
    m["face"] = True
    if fr is not None:
        w = fr["box"][2]; d = (np.linalg.norm(fc["lm"] - fr["lm"], axis=1) / w)[:3]; m["lm_err"] = float(d.max()); m["lm_err_each"] = np.round(d, 3).tolist()       # eyes + nose tip only: YuNet's mouth corners on the blocky reference are not trustworthy
        mouth = np.array(GT[gt]["mouth"], float) if gt and gt in GT else fr["lm"][3:]                       # hand-measured mouth where we have one
        if frontal: x0, x1, y0, y1 = [int(v) for v in (fr["box"][0], fr["box"][0] + fr["box"][2], mouth[:, 1].mean() - 0.12 * fr["box"][3], mouth[:, 1].mean() + 0.16 * fr["box"][3])]
        x0_, x1_, y0_, y1_ = [int(v) for v in (fr["box"][0], fr["box"][0] + fr["box"][2], fr["lm"][3:, 1].mean() - 0.12 * fr["box"][3], fr["lm"][3:, 1].mean() + 0.16 * fr["box"][3])]
        if frontal and fc is not None:
            m["mouth_corr"] = lowcorr(ref[y0:y1, x0:x1], cand[y0:y1, x0:x1], 4)
            mc = fc["lm"][3:]; m["mouth_err"] = float(np.linalg.norm(mc - mouth, axis=1).max() / w)                  # redraw's mouth corners vs the measured ones
            if m["mouth_err"] > THR["mouth_err"]: fail.append("mouth corners off by %.1f%% of face width" % (100 * m["mouth_err"]))
        if frontal and m["mouth_corr"] < THR["mouth_corr"]: fail.append("mouth strip drift (corr %.2f < %.2f)" % (m["mouth_corr"], THR["mouth_corr"]))
        # (eyes/nose landmarks are informational only: YuNet puts them unreliably on goggles and on the blocky reference)
    if frontal:
        import headtex_feat as FT
        i = FT.compare(ref, cand); m.update(gog=i["goggles"], strap=i["strap"], skin=i["skin"]); nx, nr = FT.nose_axis(cand); m["nose_dx"] = abs(nx - 512); m["nose_sym"] = nr
        for k, lab in (("gog", "goggle outline"), ("strap", "chin strap"), ("skin", "skin area")):
            if m[k] < THR[k]: fail.append("%s overlap %.2f < %.2f" % (lab, m[k], THR[k]))
        if m["nose_dx"] > THR["nose_dx"]: fail.append("nose axis %d px off the centreline" % m["nose_dx"])
        if m["nose_sym"] < THR["nose_sym"]: fail.append("crooked nose (mirror corr %.2f < %.2f)" % (m["nose_sym"], THR["nose_sym"]))
        le, re_, nose, ml, mr = fc["lm"]; w, h = fc["box"][2], fc["box"][3]                      # YuNet order: right eye, left eye, nose, right mouth corner, left mouth corner
        mid = (le + re_) / 2; m["nose_axis"] = float(abs(nose[0] - mid[0]) / w); m["mouth_tilt"] = float(abs(ml[1] - mr[1]) / h)
        mm = (ml + mr) / 2; m["mouth_axis"] = float(abs(mm[0] - mid[0]) / w)
        if m["mouth_tilt"] > THR["mouth_tilt"]: fail.append("mouth tilted %.1f%% of face height" % (100 * m["mouth_tilt"]))
    return m, fail
if __name__ == "__main__":
    from PIL import Image
    D = ROOT / "face_audit/headtex"; ref = np.array(Image.open(D / "v_cphead_front.png").convert("RGB"))
    for f in sys.argv[1:]:
        m, fl = score_view(ref, np.array(Image.open(D / f).convert("RGB")), True, "cphead_front")
        print(f, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k != "lm_err_each"}, "FAIL: " + "; ".join(fl) if fl else "PASS")
