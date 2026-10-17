#!/bin/bash
# edit.sh REF OUT "PROMPT" [extra sd-cli args...]   (Qwen-Image-2.1 edit via stable-diffusion.cpp, Vulkan)
Q=~/workspace/qwen-image; M=$Q/models
REF=$1; OUT=$2; PROMPT=$3; shift 3
exec $Q/bin/sd-cli --diffusion-model $M/${DIT:-qwen_image_2.1-Q8_0.gguf} --vae $M/qwen_image_2.1_vae_bf16.safetensors \
  --llm $M/Qwen3VL-8B-Instruct-Q4_K_M.gguf --llm_vision $M/mmproj-Qwen3VL-8B-Instruct-F16.gguf \
  -r "$REF" -p "$PROMPT" --cfg-scale ${CFG:-6.0} --sampling-method euler -W ${W:-1024} -H ${H:-1024} \
  --steps ${STEPS:-30} --seed ${SEED:-1} -o "$OUT" "$@"
