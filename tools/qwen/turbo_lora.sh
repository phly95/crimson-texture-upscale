#!/bin/bash
# turbo_lora.sh REF OUT "PROMPT" [extra args]: Q8_0 base + official Turbo LoRA, 8 steps, CFG 1 (Vulkan).
L=qwen_image_2.1_turbo_lora_avg_rank_178_bf16
REF=$1; OUT=$2; P=$3; shift 3
STEPS=8 CFG=1.0 exec ~/workspace/csvm-hd/tools/qwen/edit.sh "$REF" "$OUT" "$P <lora:$L:1>" --lora-model-dir ~/workspace/qwen-image/models --sigmas qwen-image-2.1-turbo "$@"
