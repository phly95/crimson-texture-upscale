"""hyb.py STEM ATTEMPT seeds [smooth] : hybrid (gen colours + original alpha) for each seed -> work/STEM/hyb/ATTEMPT_sN.png + sheets/STEM_hyb_ATTEMPT.png"""
import sys,os; sys.path.insert(0,'/home/deck/workspace/csvm-hd/flux_fix'); os.chdir('/home/deck/workspace/csvm-hd/flux_fix')
import alpha_tools as T
stem,att,seeds=sys.argv[1:4]; sm=float(sys.argv[4]) if len(sys.argv)>4 else 1.0
os.makedirs(f'work/{stem}/hyb',exist_ok=True); fs=[]
for s in seeds.split(','):
    o=f'work/{stem}/hyb/{att}_s{s}.png'; T.hybrid(f'work/{stem}/{att}/{os.environ.get("SRC","klein_s%s_rgba.png")%s}',stem,o,smooth=sm); fs.append(o)
T.multi(stem,fs,f'sheets/{stem}_hyb_{att}.png',int(os.environ.get('W','480')))
