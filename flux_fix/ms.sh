#!/bin/bash
# ms.sh STEM ATTEMPT [width]: contact sheet of all klein_s*_rgba.png of an attempt -> sheets/STEM_ATTEMPT.png
cd /home/deck/workspace/csvm-hd/flux_fix
../venv/bin/python -I -c "
import sys,glob; sys.path.insert(0,'.')
import alpha_tools as T
fs=sorted(glob.glob('work/$1/$2/klein_s*_rgba.png'))
T.multi('$1',fs,'sheets/$1_$2.png',int('${3:-420}'))
" 2>&1 | grep -v Warning
