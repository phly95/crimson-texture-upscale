import json, re, collections, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import sys
ROOT=Path("/home/deck/workspace/csvm-hd"); sys.path.insert(0,str(ROOT/"tools"))
import export_winners as ew
R=json.load(open(ROOT/"face_audit/faces.json")); E=np.load(ROOT/"face_audit/embs.npy"); E=E/np.linalg.norm(E,axis=1,keepdims=True)
last={}
for l in open(ROOT/"pipeline/reviews.jsonl"):
    if l.strip():
        r=json.loads(l); last[r["uid"]]=r
def stem(u): return re.sub(r"^(rimage_|scrap_|gfx_)","",u.split("__")[0]).lower()
CHAR=[("swan",r"swan"),("betty",r"betty"),("lana cooper",r"lanacooper"),("charlie",r"charlie"),("ilsa",r"ils[a]"),("nathan",r"nathan"),("zachary",r"zachary|lucas-zachary"),
      ("jack",r"(^|_)jack|-m1jack|unionjack"),("hawaiian princess",r"princes"),("treasure girl",r"treasuregirl"),("shangri-la girl",r"shangri"),("justine",r"justine"),("soldier girl",r"soldiergirl"),("initial pinup",r"initialpinup")]
def chars(s): return [n for n,p in CHAR if re.search(p,s)]
big=lambda r: r["box"][2]*r["box"][3]
sel=[r for r in R if r["kind"]=="chosen"]
bu=collections.defaultdict(list)
for r in sel: bu[r["uid"]].append(r)
def crop(uid,r,kind="chosen",sz=160):
    if kind=="orig":
        im=Image.open(ROOT/"pipeline/units"/uid/"orig.png").convert("RGB"); sc=1.0
    else:
        im=Image.open(ew.get_chosen_file(uid,last[uid]["chosen"])).convert("RGB"); sc=1.0
    x,y,w,h=r["box"]; cx,cy=x+w/2,y+h/2; s=max(w,h)*1.5
    return im.crop((int(cx-s/2),int(cy-s/2),int(cx+s/2),int(cy+s/2))).resize((sz,sz),Image.LANCZOS)
out=[]
groups=collections.defaultdict(list)
for uid,fl in bu.items():
    s=stem(uid); cs=chars(s)
    if len(cs)==1 and len(fl)>=1: groups[cs[0]].append((uid,max(fl,key=big)))
rep={}
for c,items in groups.items():
    V=np.array([E[r["emb_row"]] for _,r in items]); S=V@V.T; n=len(items)
    mean=(S.sum(1)-1)/max(n-1,1)
    med=int(np.argmax(mean)); 
    rep[c]=[(items[i][0],float(S[i,med]),float(mean[i])) for i in range(n)]
    print(f"{c}: {n} faces, mean pair cosine {(S.sum()-n)/max(n*(n-1),1):.2f}, medoid {stem(items[med][0])}")
    sheet=Image.new("RGB",(n*170 if n<=8 else 8*170,((n+7)//8)*190),(30,30,30)); d=ImageDraw.Draw(sheet)
    for k,(uid,r) in enumerate(items):
        sheet.paste(crop(uid,r),((k%8)*170,(k//8)*190)); d.text(((k%8)*170+2,(k//8)*190+162),f"{stem(uid)[:22]} {S[k,med]:.2f}",fill=(255,255,0))
    sheet.save(ROOT/f"face_audit/char_{c.replace(' ','_')}.png")
json.dump(rep,open(ROOT/"face_audit/char_report.json","w"),indent=1)
