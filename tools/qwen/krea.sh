#!/bin/bash
# krea.sh INIT_IMG OUT "PROMPT" STRENGTH [extra sd-cli args]   Krea 2 Turbo img2img via stable-diffusion.cpp (Vulkan); W/H/SEED/STEPS from the environment
K=~/workspace/krea2/models; B=~/workspace/qwen-image/bin
IN=$1; OUT=$2; PROMPT=$3; STR=$4; shift 4
exec $B/sd-cli --diffusion-model $K/${DIT:-Krea-2-Turbo-Q6_K.gguf} --vae $K/wan_2.1_vae.safetensors --llm $K/Qwen3VL-4B-Instruct-Q8_0.gguf \
  -i "$IN" -p "$PROMPT" --strength $STR --cfg-scale ${CFG:-1.0} --sampling-method euler --steps ${STEPS:-8} -W ${W:-1024} -H ${H:-1024} --seed ${SEED:-1} -o "$OUT" "$@"
