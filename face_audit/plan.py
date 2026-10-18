import json, re, collections, numpy as np
from pathlib import Path
ROOT=Path("/home/deck/workspace/csvm-hd")
R=json.load(open(ROOT/"face_audit/faces.json")); E=np.load(ROOT/"face_audit/embs.npy"); E=E/np.linalg.norm(E,axis=1,keepdims=True)
def stem(u): return re.sub(r"^(rimage_|scrap_|gfx_)","",u.split("__")[0]).lower()
rows=collections.defaultdict(lambda: collections.defaultdict(lambda: {"orig":[],"chosen":[]}))
for r in R: rows[stem(r["uid"])][r["uid"]][r["kind"]].append(r)
def fid(d):
    """per face (left to right) cos(orig, chosen)"""
    o=sorted(d["orig"],key=lambda r:r["box"][0]); c=sorted(d["chosen"],key=lambda r:r["box"][0])
    if len(o)!=len(c) or not o: 
        if o and c:
            o=[max(o,key=lambda r:r["box"][2])]; c=[max(c,key=lambda r:r["box"][2])]
        else: return []
    return [float(E[a["emb_row"]]@E[b["emb_row"]]) for a,b in zip(o,c)]
plan=[]
for s,units in rows.items():
    us=[u for u,d in units.items() if d["chosen"] and max(r["box"][2] for r in d["chosen"])>=40]
    if len(us)<2: continue
    F={u:fid(units[u]) for u in us}
    # consistency among chosen faces
    def ch(u): 
        c=sorted(units[u]["chosen"],key=lambda r:r["box"][0]); return c
    mins=[]
    for i in range(len(us)):
        for j in range(i+1,len(us)):
            a,b=ch(us[i]),ch(us[j])
            if len(a)!=len(b): a=[max(a,key=lambda r:r["box"][2])]; b=[max(b,key=lambda r:r["box"][2])]
            mins.append(min(float(E[x["emb_row"]]@E[y["emb_row"]]) for x,y in zip(a,b)))
    plan.append((min(mins),s,{u.split("__")[0][:6]+u[-4:]:[round(x,2) for x in F[u]] for u in us}))
plan.sort()
for p in plan:
    if p[0]<0.5: print(round(p[0],2),p[1],p[2])
