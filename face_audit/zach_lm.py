import json, numpy as np
content=[((75,310),(100,400)),((385,565),(470,440)),((130,545),(210,545)),((415,690),(545,510)),((300,430),(345,345)),((300,705),(330,625))]
src=np.array([a for a,b in content],float); dst=np.array([b for a,b in content],float)
A,*_=np.linalg.lstsq(np.c_[src,np.ones(len(src))],dst,rcond=None)
C=np.array([325.,255.]); r=250.
edge=[(88,68),(200,125),(300,215),(380,300),(440,380),(470,470),(465,540),(62,170),(60,250)]
out=list(content)
for p in edge:
    t=np.r_[p,1]@A; d=t-C; q=C+d/np.linalg.norm(d)*r
    out.append((p,tuple(np.round(q,1)))); print(p,'->aff',np.round(t),'->circle',np.round(q))
json.dump([[list(a),list(b)] for a,b in out],open('face_audit/zach/lm.json','w'))
