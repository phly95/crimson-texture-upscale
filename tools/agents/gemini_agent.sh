#!/bin/bash
# gemini_agent.sh [-m MODEL] [-d WORKDIR] [-t MAX_SECONDS] [-s SESSION] PROMPT|-f PROMPT_FILE
# Runs one headless Claude Code agent whose model is Gemini via the local 9router (Anthropic-compatible
# endpoint). Same setup as the nrmux alias, but non-interactive: the agent works in WORKDIR with all tools
# (Bash, Read incl. images, Edit, Write), prints its final message to stdout, and exits.
# -s SESSION: a named, resumable agent. First call with a new name starts it; every later call with the same name (and same -d) RESUMES the same
# conversation with the new prompt as the next user message, so feedback reaches the agent with its full context.
# Key: $NINEROUTER_API_KEY, else the default baked into nrmux in ~/.bashrc.
MODEL=ag/gemini-3.8-flash-medium; WORKDIR=$PWD; TMAX=1800; PROMPT=; SESSION=
while getopts "m:d:t:f:s:" o; do case $o in
  m) MODEL=$OPTARG;; d) WORKDIR=$OPTARG;; t) TMAX=$OPTARG;; f) PROMPT=$(cat "$OPTARG");; s) SESSION=$OPTARG;; esac; done
shift $((OPTIND-1)); [ -z "$PROMPT" ] && PROMPT=$*
[ -n "$PROMPT" ] || { echo "usage: gemini_agent.sh [-m model] [-d dir] [-t secs] [-f file] prompt" >&2; exit 2; }
URL=${NINEROUTER_URL:-http://100.117.72.11:20128}
KEY=${NINEROUTER_API_KEY:-$(grep -o 'sk-[0-9a-f]\{16\}-[a-z0-9]*-[0-9]*' ~/.bashrc | head -1)}
cd "$WORKDIR" || exit 1
WORKDIR=$(pwd -P)
SESS=(--no-session-persistence)
if [ -n "$SESSION" ]; then
  SID=$(python3 -c "import uuid,sys;print(uuid.uuid5(uuid.NAMESPACE_URL,sys.argv[1]+sys.argv[2]))" "$SESSION" "$WORKDIR")
  mkdir -p ~/.cache/gemini_agent; M=~/.cache/gemini_agent/$SID
  if [ -f "$M" ]; then SESS=(--resume "$SID"); else SESS=(--session-id "$SID"); touch "$M"; echo "[gemini_agent] new session $SESSION ($SID)" >&2; fi
fi
ANTHROPIC_BASE_URL=$URL ANTHROPIC_AUTH_TOKEN=$KEY ANTHROPIC_API_KEY=$KEY \
ANTHROPIC_MODEL=$MODEL ANTHROPIC_SMALL_FAST_MODEL=$MODEL \
ANTHROPIC_DEFAULT_OPUS_MODEL=$MODEL ANTHROPIC_DEFAULT_SONNET_MODEL=$MODEL ANTHROPIC_DEFAULT_HAIKU_MODEL=$MODEL \
CLAUDE_CODE_MAX_CONTEXT_TOKENS=1048576 \
timeout "$TMAX" claude -p "$PROMPT" --dangerously-skip-permissions "${SESS[@]}" 2>&1
