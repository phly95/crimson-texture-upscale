import numpy as np, cv2, sys
from PIL import Image
a=np.asarray(Image.open('output.png').convert('RGBA')).astype(np.float32)/255
def comp(bg):
    bg=np.array(bg,np.float32)/255
    return (a[...,:3]*a[...,3:]+bg*(1-a[...,3:]))
g=comp((210,210,210)); k=comp((0,170,0)); d=comp((20,20,20))
sm=[cv2.resize(v,(682,682),interpolation=cv2.INTER_AREA) for v in (g,d,k)]
Image.fromarray((np.concatenate(sm,1)*255).astype(np.uint8)).save('check_view.png')
def crop(v,x,y,w,h,z): return cv2.resize(v[y:y+h,x:x+w],(w*z,h*z),interpolation=cv2.INTER_NEAREST)
sc='/tmp/claude-1000/-home-deck/8d33977a-9b65-41a7-af0e-5ee75413ed5e/scratchpad/'
regs={'e1':(0,230,400,100,2),'e2':(300,1000,300,200,3),'e3':(900,230,300,150,3),'e4':(1500,1350,350,150,3),'e5':(100,1380,350,120,3)}
for n,(x,y,w,h,z) in regs.items():
    c=np.concatenate([crop(g,x,y,w,h,z),crop(k,x,y,w,h,z)],0)
    Image.fromarray((c*255).astype(np.uint8)).save(sc+n+'.png')
