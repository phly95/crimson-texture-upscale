"""mag_comic_fix.py: pass-2 clean-up of the comic cover SB_22_01_COMICOVER (called by mag_finish.py; W is the prefix of the intermediate files).
  a  the title: the redraw's bevel/emboss shading, glow and orange drop-shadow 'drips': the letters (orange components fully ringed by their dark outline, gated by the
     original's orange) are set in the original's flat orange, the outline in one dark, the glow / shadow round them refilled from the cream paper (ladder rails kept)
  b  'Fortune Hunter!': the redraw's line was garbled (Hunrter) and embossed: wiped and set again in Comic Sans along the original's line; a stray rung-end dash removed
  c  broad grey shading left in the title's paper (shadows by the ladder) flattened to the paper colour (thin line art is never caught: morphological opening)
  e  statue colour lock (class-wise) and the grey backing paper edge
  d  the statue's green: the redraw's dark vignettes along the lines; a colour lock whose low frequencies leave the line art out on both sides (no halo)"""
import sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mag_polish as mp


def run(src, W):
    stem = 'SB_22_01_COMICOVER'
    arr=np.asarray(Image.open(src)); rgb=arr[...,:3]; f=rgb.astype(int)
    H,Wd=rgb.shape[:2]
    reg=mp.poly_mask(arr.shape,[[(540,0),(1444,0),(1444,412),(470,412),(470,292),(540,292)]])   # title area (big letters + Fortune Hunter!)
    R,G,B=f[...,0],f[...,1],f[...,2]
    orange=(R-B>70)&(R>150)&(R-G>50)&reg
    o1=np.asarray(Image.open(mp.SB/(stem+'.PNG')).convert('RGB')).astype(int)
    g1=((o1[...,0]-o1[...,2]>70)&(o1[...,0]>150)).astype(np.float32)
    gate=cv2.resize(g1,(Wd,H),interpolation=cv2.INTER_LINEAR)>0.3
    gate=cv2.dilate(gate.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))>0
    orange&=gate
    orange=cv2.morphologyEx(orange.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))>0
    n,lab,st,_=cv2.connectedComponentsWithStats(orange.astype(np.uint8),8)
    Lg=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    dk=Lg<100
    big=np.zeros(n,bool)
    for i in range(1,n):
        if st[i,4]<400: continue
        x,y,w,h=st[i,:4]; x0,y0=max(x-6,0),max(y-6,0)
        c=(lab[y0:y+h+6,x0:x+w+6]==i).astype(np.uint8)
        ring=(cv2.dilate(c,np.ones((7,7),np.uint8))>0)&(c==0)
        fr=dk[y0:y+h+6,x0:x+w+6][ring].mean()
        big[i]=fr>0.6
    orange=big[lab]
    # fill small holes inside letters (bevel highlights)
    orange=cv2.morphologyEx(orange.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))>0
    L=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    near=cv2.dilate(orange.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))>0
    dark=(L<95)&near&~orange
    outline=dark
    ft=mp.poly_mask(arr.shape,[[(1000,330),(1444,330),(1444,470),(1000,470)]])&(L<110)
    halo=(cv2.dilate(orange.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(71,71)))>0)|(cv2.dilate(ft.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41)))>0)
    orng=(R-B>50)&(R>140)&reg&~orange                              # the redraw's orange drop shadow / drips not in the original
    halo=(halo|orng)&reg&~orange&~outline
    blue=(B>R+15)&(L<205)                                   # ladder rails and other bluish-grey art stay
    art=(L<110)                                            # any other line art stays
    prot=cv2.dilate((blue|art).astype(np.uint8),np.ones((3,3),np.uint8))>0
    prot&=~orng
    todo=halo&~prot
    cream=(f.sum(-1)>600)&(R-B>15)&~near
    keep=cream&~todo
    fill=mp._fill(rgb.astype(np.float32),keep,todo,(4,10,30,80))
    fill+=np.random.default_rng(3).normal(0,1.6,(H,Wd))[...,None]
    out=rgb.astype(np.float32)
    w=cv2.GaussianBlur(todo.astype(np.float32),(0,0),1.2)[...,None]
    out=out*(1-w)+fill*w
    # flat letters and outline
    ao=cv2.GaussianBlur(orange.astype(np.float32),(0,0),0.8)[...,None]
    out=out*(1-ao)+np.array((229,99,37),np.float32)*ao
    ad=np.clip((110-L)/50,0,1)*outline
    out=out*(1-ad[...,None])+np.array((30,26,24),np.float32)*ad[...,None]
    Image.fromarray(np.dstack([out.clip(0,255).astype(np.uint8),arr[...,3]])).save(W+'a.png')

    cream=lambda r: (r.astype(int).sum(-1)>560)&(r[...,0].astype(int)-r[...,2].astype(int)>12)
    mp.fill_poly(W+'a.png', W+'a3.png', [(870,330),(1395,350),(1395,452),(1030,425),(950,405),(870,400)], cream, sig=(4,10,30,80))
    A=np.asarray(Image.open(W+'a.png')); B=np.asarray(Image.open(W+'a3.png'))
    f=A[...,:3].astype(int); org=(f[...,0]-f[...,2]>70)&(f[...,0]>150)
    keepo=cv2.dilate(org.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)))>0
    keepo&=~((f.sum(-1)>560)&(f[...,0]-f[...,2]>12))   # the letters and their outline only, not the paper round them
    keepo=cv2.dilate(keepo.astype(np.uint8),np.ones((3,3),np.uint8))>0
    Image.fromarray(np.where(keepo[...,None],A,B)).save(W+'a3.png')
    mp.tilted_text(W+'a3.png', W+'b.png', "Fortune Hunter!", 878, 397, 1358, 0.073, 52, mp.WIN/'COMIC.TTF', (48,50,52), cream, rect_up=1.25, rect_down=0.12, soft=0.7, space=1.0)
    Bb=np.asarray(Image.open(W+'b.png')); A3=np.asarray(Image.open(W+'a3.png'))
    txt=(A3[...,:3].astype(int).sum(-1)-Bb[...,:3].astype(int).sum(-1))>90
    txt=cv2.dilate(txt.astype(np.uint8),np.ones((5,5),np.uint8))>0
    hair=mp.poly_mask(A.shape,[[(812,450),(824,398),(840,386),(862,381),(885,379),(905,379),(930,386),(960,396),(1000,418),(1010,450)]])   # the hair tuft that rises
    # into the old line's box: the previous pixels (the wipe had cut its strands off); the new letters stay on top
    tA=((A3[...,:3].astype(int).sum(-1)-Bb[...,:3].astype(int).sum(-1))>30)            # the new letters' own pixels (soft edge included)
    hm=cv2.GaussianBlur((hair&~tA).astype(np.float32),(0,0),0.8)*(1-cv2.GaussianBlur(tA.astype(np.float32),(0,0),0.6))
    res=np.where((keepo&~txt)[...,None],A,Bb).astype(np.float32)
    res=res*(1-hm[...,None])+A.astype(np.float32)*hm[...,None]
    Image.fromarray(res.clip(0,255).astype(np.uint8)).save(W+'b.png')
    mp.fill_poly(W+'b.png', W+'b.png', [(927,384),(938,384),(938,395),(927,395)], cream, sig=(3,8,20))      # an old stroke's tip inside the new 'o'
    for cx,cy in ((1172,360),(1359,383),(1140,352)):
        mp.fill_poly(W+'b.png', W+'b.png', [(cx-6,cy-7),(cx+6,cy-7),(cx+6,cy+7),(cx-6,cy+7)], cream, sig=(3,8,20))

    mp.fill_poly(W+'b.png', W+'b.png', [(872,316),(909,316),(909,334),(872,334)], cream, sig=(3,8,20))
    mp.fill_poly(W+'b.png', W+'b.png', [(1166,364),(1176,364),(1176,371),(1166,371)], cream, sig=(3,8,20))     # a speck above the 'H'
    mp.fill_poly(W+'b.png', W+'b.png', [(533,242),(552,241),(546,262),(533,262)], cream, sig=(3,8,20))     # a dark blob of the old extrusion after 'GOLD'
    bb=np.asarray(Image.open(W+'b.png')); fb=bb[...,:3].astype(int)                                       # grey strokes of the old line over the small 'Z' of ZACHARY
    gm=np.zeros(fb.shape[:2],bool); gm[436:464,1119:1137]=True
    gm&=(fb[...,0]-fb[...,2]<60)&(fb.mean(-1)<200)
    gm=cv2.dilate(gm.astype(np.uint8),np.ones((3,3),np.uint8))>0
    okc=(fb.sum(-1)>560)&(fb[...,0]-fb[...,2]>12)&~gm
    gf=mp._fill(fb.astype(np.float32),okc,gm,(2,4,8,20))
    Image.fromarray(np.dstack([np.where(gm[...,None],gf,fb).clip(0,255).astype(np.uint8),bb[...,3]])).save(W+'b.png')
    arr=np.asarray(Image.open(W+'b.png')); rgb=arr[...,:3]; f=rgb.astype(int); H,Wd=rgb.shape[:2]
    reg=mp.poly_mask(arr.shape,[[(540,0),(1444,0),(1444,330),(1060,300),(800,330),(560,300)]])
    cream=(f.sum(-1)>600)&(f[...,0]-f[...,2]>15)
    Lr=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    cf=mp._fill(rgb.astype(np.float32),cream,~cream,(6,15,40,100))
    Lc=cv2.cvtColor(cf.clip(0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)
    orange=(f[...,0]-f[...,2]>60)&(f[...,0]>140)
    off=(np.abs(Lr-Lc)>6)|(np.abs(f[...,2]-cf[...,2])>10)
    broad=cv2.morphologyEx((off&~orange).astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,)*2))>0
    nearl=cv2.dilate((orange|(Lr<100)).astype(np.uint8),np.ones((5,5),np.uint8))>0
    todo=broad&reg&~nearl&(Lr>110)
    w=cv2.GaussianBlur(todo.astype(np.float32),(0,0),2)[...,None]
    out=rgb*(1-w)+(cf+np.random.default_rng(4).normal(0,1.6,(H,Wd))[...,None])*w
    Image.fromarray(np.dstack([out.clip(0,255).astype(np.uint8),arr[...,3]])).save(W+'c.png')

    arr=np.asarray(Image.open(W+'c.png')); f=arr[...,:3].astype(int)
    ref=mp.ref_page(stem,4).astype(int)
    green=lambda r: (r[...,1]-r[...,0]>15)&(r[...,1]-r[...,2]>30)&(r.mean(-1)>90)
    gr=cv2.erode(green(ref).astype(np.uint8),np.ones((9,9),np.uint8))>0       # the original's flat green, away from its (blurred) lines
    gf=green(f)&(f.mean(-1)>100)
    gf=cv2.erode(gf.astype(np.uint8),np.ones((3,3),np.uint8))>0
    region=cv2.GaussianBlur((cv2.dilate(green(ref).astype(np.uint8),np.ones((9,9),np.uint8))>0).astype(np.float32),(0,0),4)*green(f)
    mp.lock_flat(stem, W+'c.png', W+'d.png', region, gf, gr, sig=12.0)

    # e  the statue: washed, glowing gradient and pale halos along the lines: a colour + brightness lock computed within the statue's own colour class on both sides
    #    (lines, cream and grey paper never enter the averages, so it does not halo); the grey backing paper left of the arm repainted up to the original's torn edge
    dst=W+'e.png'
    sig=6.0
    arr=np.asarray(Image.open(W+'d.png')); f=arr[...,:3].astype(int)
    ref=mp.ref_page(stem,4).astype(int)
    def statue(r):
        L=r.mean(-1)
        return (L>85)&(((r[...,1]-r[...,0])>6)|((r[...,2]-r[...,0])>6))&~((np.abs(r[...,0]-r[...,1])<8)&(np.abs(r[...,1]-r[...,2])<8))
    sr=cv2.erode(statue(ref).astype(np.uint8),np.ones((5,5),np.uint8))>0
    sf=statue(f)
    reg=(cv2.dilate(statue(ref).astype(np.uint8),np.ones((15,15),np.uint8))>0)&sf
    reg&=~mp.poly_mask(reg.shape,[[(990,280),(1444,280),(1444,490),(990,490)]])      # never the cover-line text box
    reg=cv2.GaussianBlur(reg.astype(np.float32),(0,0),1.0)
    mp.lock_flat(stem, W+'d.png', dst, reg, sf, sr, sig=sig)
    # the grey backing paper left of the statue: the redraw blurred the green into it; repaint the paper side of the original's edge with the paper's own grey
    o1=np.asarray(Image.open(mp.SB/(stem+'.PNG')).convert('RGB')).astype(int)
    g1=((np.abs(o1[...,0]-o1[...,1])<14)&(np.abs(o1[...,1]-o1[...,2])<14)&(o1.mean(-1)>140)).astype(np.float32)
    H,Wi=f.shape[:2]
    pg=cv2.resize(cv2.GaussianBlur(g1,(0,0),0.7),(Wi,H),interpolation=cv2.INTER_CUBIC)>0.5
    box=np.zeros((H,Wi),bool); box[600:1684,0:700]=True
    pg&=box
    a2=np.asarray(Image.open(dst)); f2=a2[...,:3].astype(int)
    grey=(np.abs(f2[...,0]-f2[...,1])<10)&(np.abs(f2[...,1]-f2[...,2])<10)&(f2.mean(-1)>150)&pg
    grey=cv2.erode(grey.astype(np.uint8),np.ones((5,5),np.uint8))>0
    todo=pg&~grey
    fill=mp._fill(f2.astype(np.float32),grey,todo,(4,10,30,80))+np.random.default_rng(6).normal(0,1.4,(H,Wi))[...,None]
    w=cv2.GaussianBlur(todo.astype(np.float32),(0,0),0.8)[...,None]
    out=f2*(1-w)+fill*w
    Image.fromarray(np.dstack([out.clip(0,255).astype(np.uint8),a2[...,3]])).save(dst)

    return W+'e.png'
