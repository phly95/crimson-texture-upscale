#!/bin/bash
# retry_until_ok.sh WORKDIR SESSION PROMPTFILE [MAX_MINUTES]: run a (resumable) Gemini agent turn; while 9router/Gemini answers with a quota or availability error
# (API Error 429/503), wait 5 minutes and try again. Prints the agent's answer when one arrives.
D=$1; S=$2; P=$3; MAX=${4:-240}; end=$(( $(date +%s) + MAX*60 ))
while [ "$(date +%s)" -lt "$end" ]; do
  out=$("$(dirname "$0")/gemini_agent.sh" -d "$D" -s "$S" -t 3600 -f "$P" </dev/null 2>&1)
  if echo "$out" | grep -q "API Error: \(429\|503\)"; then echo "[$(date +%H:%M)] quota/unavailable, retrying in 5 min: $(echo "$out" | grep -o 'reset after [0-9hm ]*' | head -1)"; sleep 300; continue; fi
  echo "$out"; exit 0
done
echo "gave up after $MAX minutes of quota errors"; exit 1
