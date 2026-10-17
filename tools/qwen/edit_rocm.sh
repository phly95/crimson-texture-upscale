#!/bin/bash
# edit_rocm.sh REF OUT "PROMPT" [sd-cli args...]  - same as edit.sh but on the ROCm backend inside the llama-rocm container.
# REF/OUT are paths relative to ~/workspace/csvm-hd/qwen_test (mounted at /t).
Q=/q/models; REF=$1; OUT=$2; PROMPT=$3; shift 3
exec podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt label=disable \
  -v $HOME/workspace/qwen-image:/q:ro -v $HOME/workspace/csvm-hd/qwen_test:/t --workdir /t --entrypoint /q/bin-rocm/build/bin/sd-cli \
  localhost/llama-rocm:b11580 \
  --diffusion-model $Q/${DIT:-qwen_image_2.1-Q8_0.gguf} --vae $Q/qwen_image_2.1_vae_bf16.safetensors \
  --llm $Q/Qwen3VL-8B-Instruct-Q4_K_M.gguf --llm_vision $Q/mmproj-Qwen3VL-8B-Instruct-F16.gguf \
  -r "$REF" -p "$PROMPT" --cfg-scale ${CFG:-6.0} --sampling-method euler -W ${W:-1024} -H ${H:-1024} \
  --steps ${STEPS:-30} --seed ${SEED:-1} -o "$OUT" "$@"
