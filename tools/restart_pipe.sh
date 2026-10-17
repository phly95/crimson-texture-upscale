#!/bin/bash
# restart_pipe.sh: stop the pipeline after the unit it is on, then start it again (tmux session csvm-pipe)
cd /home/deck/workspace/csvm-hd
n0=$(grep -c "pipeline ended" pipeline/run.log)
touch pipeline/STOP
until [ "$(grep -c 'pipeline ended' pipeline/run.log)" -gt "$n0" ]; do sleep 5; done
sleep 3
tmux kill-session -t csvm-pipe 2>/dev/null
rm -f pipeline/STOP
echo "--- run restarted $(date +%H:%M:%S)" >> pipeline/run.log
STMUX_NOATTACH=1 bash -ic 'stmux csvm-pipe "cd /home/deck/workspace/csvm-hd && venv/bin/python -I tools/pipeline/run.py 2>&1 | tee -a pipeline/run.out"'
echo restarted
