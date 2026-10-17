#!/bin/bash
# gemma_server.sh [NP]  : Gemma 4 E4B-it Q8_0 + vision on llama.cpp ROCm (b11580), port 9933, NP slots (default 8) x 16k context
NP=${1:-8}
exec podman run --rm --replace --name llm-gemma --network host --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt label=disable --ipc=host -v ~/workspace/gemma4/models:/models:ro localhost/llama-rocm:b11580 \
  -m /models/gemma-4-E4B-it-Q8_0.gguf --mmproj /models/mmproj-gemma-4-E4B-it-BF16.gguf \
  --host 127.0.0.1 --port 9933 -ngl 99 -fa on -np $NP -c $((NP*16384)) ${EXTRA}
