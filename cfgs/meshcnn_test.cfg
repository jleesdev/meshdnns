#!/usr/bin/env bash

## run the test and export collapses
python test.py \
--dataroot datasets/adni2_clf/dp2191_ac141_f/ \
--name dp2191_ac141_lr5e-4_bs16_aug0.2_adam_ep300 \
--ncf 16 32 64 128 \
--pool_res 1200 720 450 300 \
--fc_n 8 \
--norm group \
--resblocks 1 \
--export_folder meshes \
--gpu_ids 1 \
--ninput_edges 2190 \
--phase train \