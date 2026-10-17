#!/bin/bash
# Krea 2 Turbo (Q6_K GGUF) + Wan2.1 VAE + Qwen3-VL-4B text encoder (Q8_0) for stable-diffusion.cpp; sha256 prefix checked against HF LFS oids.
cd ~/workspace/krea2/models || exit 1
get() { until curl -fL --retry 5 --retry-delay 5 -C - -o "$2" "$1"; do sleep 5; done
  h=$(sha256sum "$2" | cut -c1-12); [ "$h" = "$3" ] && echo "OK $2" || echo "HASH MISMATCH $2 got $h want $3"; }
get https://huggingface.co/realrebelai/KREA-2_GGUFs/resolve/main/TURBO/Krea-2-Turbo-Q6_K.gguf Krea-2-Turbo-Q6_K.gguf d95c232c7835
get https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors wan_2.1_vae.safetensors 2fc39d31359a
get https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3VL-4B-Instruct-Q8_0.gguf Qwen3VL-4B-Instruct-Q8_0.gguf 054721f478bc
echo ALL_DONE
