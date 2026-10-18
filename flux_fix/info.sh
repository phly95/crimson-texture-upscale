#!/bin/bash
# info.sh STEM : print hint+prompts+review and make the old-redraw alpha sheet
cd /home/deck/workspace/csvm-hd
S=$1; U=pipeline/units/$(venv/bin/python -I -c "
import sys; sys.path.insert(0,'flux_fix')
import alpha_tools as T; print(T.unit_of('$S'))" 2>/dev/null)
cat $U/hint.txt; echo; echo --PROMPT; cat $U/prompt.txt | tail -2; echo --RETRY; cat $U/retry/prompt.txt 2>/dev/null | tail -2; echo --REVIEW; head -3 $U/review.txt
venv/bin/python -I flux_fix/alpha_tol.py $S pack_cache/$(basename $U).png flux_fix/sheets/${S}_old.png 2>&1 | grep -v Warn
