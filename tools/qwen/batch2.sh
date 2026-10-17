#!/bin/bash
cd ~/workspace/csvm-hd/qwen_test
P2="This is an RGBA image with transparency. Redraw the reference low-resolution game decal as a crisp, highly detailed 1930s pulp-comic nose-art sticker illustration: a snarling anthropomorphic skunk pilot with black fur, a white stripe running up his forehead, a long snout with bared teeth, a leather flight cap with brass goggles, blue denim overalls with rivets, orange work gloves, carrying a tommy gun with a round drum magazine and a wooden rifle slung over his back, his huge fluffy black-and-white striped skunk tail curling up behind him, under the arched glossy gold bubble lettering 'BIG STINK' with white sparkle stars. Keep exactly the same composition, pose and colour palette as the reference; clean black ink outlines, detailed fur texture, worn chipped paint, rust stains and grime, and a cream sticker border. The image has alpha channel and the background is transparent."
# A: base model, 40 steps, CFG 4
STEPS=40 CFG=4.0 ../tools/qwen/edit.sh in/bigstink_1024.png out/bs2_base40_cfg4.png "$P2" > out/bs2_base40_cfg4.log 2>&1
# B: turbo-lora, three seeds
for s in 2 3 4; do SEED=$s ../tools/qwen/turbo_lora.sh in/bigstink_1024.png out/bs2_turbo_seed$s.png "$P2" > out/bs2_turbo_seed$s.log 2>&1; done
echo done > out/batch2.done
