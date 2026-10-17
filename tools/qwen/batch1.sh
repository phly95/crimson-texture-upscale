#!/bin/bash
cd ~/workspace/csvm-hd/qwen_test
E=../tools/qwen/edit.sh
$E in/skull_1024.png out/skull_generic_q8.png "Redraw the reference low-resolution game texture as a crisp, highly detailed high-resolution black and white ink illustration. Keep exactly the same composition, subject, shading and the black background; add clean brush linework and fine detail." > out/skull_generic_q8.log 2>&1
$E in/farm_1024.png out/farm_generic_q8.png "Upscale the reference aerial photograph to high resolution. Keep exactly the same field layout, colours, tree lines and paths; add realistic fine detail: crop rows, ploughed furrows, individual trees, subtle film grain. Aerial photograph." > out/farm_generic_q8.log 2>&1
echo BATCH_DONE > out/batch1.done
