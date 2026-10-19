"""zach_guides.py: the user's two red baseline lines (drawn on a zoomed OG crop) -> anchor-window coordinates; overlay on candidates"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/deck/workspace/csvm-hd/face_audit")
import numpy as np, cv2, regen as R
from PIL import Image
Z = "/home/deck/workspace/csvm-hd/face_audit/zach/"
# screenshot px <-> 6b OG-crop grid px (6x crop at (50,75))
S_pts = np.array([(200, 285), (500, 770), (655, 1110)], float); G_pts = np.array([(88, 68), (300, 430), (415, 690)], float)
ax = np.polyfit(G_pts[:, 0], S_pts[:, 0], 1); ay = np.polyfit(G_pts[:, 1], S_pts[:, 1], 1)
print("screen = %.2f*grid%+.0f , %.2f*grid%+.0f" % (ax[0], ax[1], ay[0], ay[1]))
L1 = np.array([(165, 750), (290, 805), (450, 915), (640, 1025)], float)       # Fortune baseline (the user's red line, polyline)
L2 = np.array([(190, 900), (400, 1020), (680, 1190)], float)                   # Hunters baseline
def to_grid(P): return np.c_[(P[:, 0] - ax[1]) / ax[0], (P[:, 1] - ay[1]) / ay[0]]
f = lambda s: [u for u in R.last if u.endswith(s)][0]
Hog, n = R.homography(f("3ad54545"), f("6b4b3dd2")); Hinv = np.linalg.inv(Hog)
def to_anchor_result(G):
    og6 = G / 6.0 + np.array([50, 75]); h = np.c_[og6, np.ones(len(og6))] @ Hinv.T; oga = h[:, :2] / h[:, 2:3]
    return oga * 4.0                                                           # anchor result px (1024)
OUT = np.array([(190, 290), (300, 320), (430, 400), (540, 500), (640, 620), (700, 720), (725, 830), (715, 900), (680, 960), (600, 1010), (480, 1130), (340, 1110), (230, 1010), (185, 840), (165, 600), (172, 400)], float)   # the oval edge in the same screenshot
def outline_in_anchor(): return to_anchor_result(to_grid(OUT))
def lines_in_anchor(): return [to_anchor_result(to_grid(L)) for L in (L1, L2)]
def window(P): return (P - np.array([0, 40])) * (1024 / 540)                    # anchor result px -> klein window (0,40,540,580) at 1024
def angle(P):
    d = P[-1] - P[0]; return np.degrees(np.arctan2(d[1], d[0]))
if __name__ == "__main__":
    A = lines_in_anchor(); print("baseline angles in the anchor picture:", [round(angle(p), 1) for p in A])
    print("gap between baselines (anchor px):", round(float(np.linalg.norm(A[0][1:3].mean(0) - A[1][:2].mean(0))), 1))
    def overlay(imgpath, out):
        im = cv2.imread(imgpath); w = [window(p).astype(np.int32) for p in A]
        for p in w: cv2.polylines(im, [p.reshape(-1, 1, 2)], False, (0, 0, 255), 3)
        o = window(outline_in_anchor()).astype(np.int32); cv2.polylines(im, [o.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        cv2.imwrite(out, im)
    for t in sys.argv[1:]: overlay(Z + t, Z + "g_" + t); print("wrote g_" + t)
