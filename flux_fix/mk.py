"""mk.py STEM NAME [mode thr smooth [r,g,b]] : write guides/STEM_NAME.png"""
import sys; sys.path.insert(0,'/home/deck/workspace/csvm-hd/flux_fix')
import alpha_tools as T
stem,name=sys.argv[1:3]; mode=sys.argv[3] if len(sys.argv)>3 else 'hard'
thr=float(sys.argv[4]) if len(sys.argv)>4 else .5; sm=float(sys.argv[5]) if len(sys.argv)>5 else 2.0
bg=tuple(int(v) for v in sys.argv[6].split(',')) if len(sys.argv)>6 else None
T.guide(stem,mode,thr=thr,smooth=sm,bg=bg).save(f'/home/deck/workspace/csvm-hd/flux_fix/guides/{stem}_{name}.png')
