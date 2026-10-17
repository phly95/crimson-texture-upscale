#!/bin/bash
# klein.sh 4b|9b REF OUT "PROMPT" [extra sd-cli args...]   (FLUX.2 klein distilled edit via stable-diffusion.cpp, Vulkan)
K=~/workspace/flux2-klein/models; B=~/workspace/qwen-image/bin
SZ=$1; REF=$2; OUT=$3; PROMPT=$4; shift 4
case $SZ in 4b) LLM=Qwen3-4B-Q8_0.gguf;; 9b) LLM=Qwen3-8B-Q8_0.gguf;; esac
exec $B/sd-cli --diffusion-model $K/flux-2-klein-$SZ-Q8_0.gguf --vae $K/flux2-vae.safetensors --llm $K/$LLM \
  -r "$REF" -p "$PROMPT" --cfg-scale 1.0 --sampling-method euler -W ${W:-1024} -H ${H:-1024} --steps ${STEPS:-4} --seed ${SEED:-1} -o "$OUT" "$@"
