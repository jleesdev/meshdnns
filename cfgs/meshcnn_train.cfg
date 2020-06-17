#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/adni2_clf/dp2191_ac141_f/ \
--dataset_mode classification \
--ninput_edges 2190 \
--name dp2191_ac141_lr5e-4_bs16_aug0.2_adam_ep300_avg \
--ncf 16 32 64 128 \
--pool_res 1200 720 450 300 \
--fc_n 8 \
--norm group \
--resblocks 1 \
--flip_edges 0.1 \
--slide_verts 0.1 \
--scale_verts true \
--lr 0.0005 \
--num_aug 5 \
--niter_decay 150 \
--batch_size 16 \
--gpu_ids 1 \
--niter 150 \
--save_epoch_freq 1 \
--optim Adam \
