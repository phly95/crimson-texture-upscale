#!/bin/bash
# Gemma 4 E4B-it (ggml-org conversions): Q8_0 model, BF16 mmproj, Q8_0 MTP drafter; sha256 checked against HF LFS oids.
cd ~/workspace/gemma4/models || exit 1
B=https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main
get() { until curl -fL --retry 5 --retry-delay 5 -C - -o "$1" "$B/$1"; do sleep 5; done
  h=$(sha256sum "$1" | cut -c1-12); [ "$h" = "$2" ] && echo "OK $1" || echo "HASH MISMATCH $1 got $h want $2"; }
get gemma-4-E4B-it-Q8_0.gguf 34be82b17b49
get mmproj-gemma-4-E4B-it-BF16.gguf f77995e4b6a5
get mtp-gemma-4-E4B-it-Q8_0.gguf f38ae6296265
echo ALL_DONE
