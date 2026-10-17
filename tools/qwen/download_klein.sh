#!/bin/bash
# Download FLUX.2 klein 4B/9B (distilled GGUF) + Qwen3 text encoders + VAE into ~/workspace/flux2-klein/models, then verify sha256 against the HF LFS oids.
cd ~/workspace/flux2-klein/models || exit 1
get() { # url name sha256-prefix
  until curl -fL --retry 5 --retry-delay 5 -C - -o "$2" "$1"; do echo "retrying $2"; sleep 5; done
  h=$(sha256sum "$2" | cut -c1-12); [ "$h" = "$3" ] && echo "OK $2" || echo "HASH MISMATCH $2 got $h want $3"
}
get https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q8_0.gguf flux-2-klein-4b-Q8_0.gguf 0bba6951258e
get https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors flux2-vae.safetensors d64f3a68e1cc
get https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf Qwen3-4B-Q8_0.gguf eed555233267
get https://huggingface.co/leejet/FLUX.2-klein-9B-GGUF/resolve/main/flux-2-klein-9b-Q8_0.gguf flux-2-klein-9b-Q8_0.gguf 67ca777c7aa2
get https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf Qwen3-8B-Q8_0.gguf 0cfbf745760f
echo ALL_DONE
