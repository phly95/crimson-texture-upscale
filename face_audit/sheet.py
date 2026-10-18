import json,sys,warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"/home/deck/workspace/csvm-hd/face_audit"); sys.path.insert(0,"/home/deck/workspace/csvm-hd/tools")
import numpy as np
from PIL import Image, ImageDraw
import transfer as T, export_winners as ew
ROOT="/home/deck/workspace/csvm-hd"
last={}
for l in open(ROOT+"/pipeline/reviews.jsonl"):
    if l.strip():
        r=json.loads(l); last[r["uid"]]=r
log=json.load(open(ROOT+"/face_audit/transfer_log.json"))
def crop(im,f,sz=240):
    x,y,w,h=f[:4]; cx,cy=x+w/2,y+h/2; s=max(w,h)*1.9; return Image.fromarray(im).crop((int(cx-s/2),int(cy-s/2),int(cx+s/2),int(cy+s/2))).resize((sz,sz),Image.LANCZOS)
rows=[]
for s,u,a,done in log:
    before=np.asarray(Image.open(ew.get_chosen_file(u,last[u]["chosen"])).convert("RGB")); after=np.asarray(Image.open(f"{ROOT}/face_audit/out/{u}.png").convert("RGB")); anc=np.asarray(Image.open(ew.get_chosen_file(a,last[a]["chosen"])).convert("RGB"))
    fb=[f for f in T.detect(before) if f[2]>=40]; fa=[f for f in T.detect(anc) if f[2]>=40]
    H,W=before.shape[:2]; Ha,Wa=anc.shape[:2]
    for fa_,ft in T.match(fa,fb,Wa,Ha,W,H):
        x,y,w,h=[int(v) for v in ft[:4]]; reg=slice(max(0,y),y+h),slice(max(0,x),x+w)
        if np.abs(before[reg].astype(int)-after[reg].astype(int)).mean()>1.5:
            rows.append((s+" "+u[-8:]+" @%.2f,%.2f"%((ft[0]+ft[2]/2)/W,(ft[1]+ft[3]/2)/H),crop(before,ft),crop(after,ft),crop(anc,fa_)))
print(len(rows),"faces")
for k in range(0,len(rows),8):
    part=rows[k:k+8]
    sh=Image.new("RGB",(3*250*2,((len(part)+1)//2)*270),(30,30,30)); d=ImageDraw.Draw(sh)
    for i,(t,b,a,c) in enumerate(part):
        X=(i%2)*750; Y=(i//2)*270
        for j,im in enumerate((b,a,c)): sh.paste(im,(X+j*250,Y))
        d.text((X+3,Y+244),t+"  before|after|anchor",fill=(255,255,0))
    sh.save(f"{ROOT}/face_audit/sheet_{k//8}.png")
