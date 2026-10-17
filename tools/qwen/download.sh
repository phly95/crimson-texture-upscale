#!/bin/bash
# Download the Qwen-Image-2.1 pieces for stable-diffusion.cpp into ~/workspace/qwen-image/models (resumable).
cd ~/workspace/qwen-image/models || exit 1
get() { # url name
  until curl -fL --retry 5 --retry-delay 5 -C - -o "$2" "$1"; do echo "retrying $2"; sleep 5; done
  echo "done $2 $(stat -c %s "$2")"
}
get https://huggingface.co/leejet/Qwen-Image-2.1-GGUF/resolve/main/qwen_image_2.1-Q8_0.gguf qwen_image_2.1-Q8_0.gguf
get https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors qwen_image_2.1_vae_bf16.safetensors
get https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3VL-8B-Instruct-Q4_K_M.gguf Qwen3VL-8B-Instruct-Q4_K_M.gguf
get https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-8B-Instruct-F16.gguf mmproj-Qwen3VL-8B-Instruct-F16.gguf
echo ALL_DONE
