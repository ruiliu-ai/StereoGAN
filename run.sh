#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
model_name="StereoGAN_maxdisp192_cycle10_id5_corr1_ms1e-1_invwarp5_invdispwarp5_imwidth512_height256_ep100_lr1e-5_gan2e-5"

srun --partition=Pixel --mpi=pmi2 --gres=gpu:4 -n1 --ntasks-per-node=4 \
--job-name=${model_name} --kill-on-bad-exit=1 python -u train.py \
--model_type='dispnetc' \
--source_dataset='driving' \
--batch_size=4 \
--test_batch_size=4 \
--lr_rate=1e-5 \
--lr_gan=2e-5 \
--train_ratio_gan=3 \
--total_epochs=101 \
--save_interval=5 \
--print_freq=220 \
--checkpoint_save_path="checkpoints/${model_name}" \
--writer=${model_name} \
--use_multi_gpu=4 \
--img_height=256 \
--img_width=512 \
--maxdisp=192 \
--lambda_corr=1 \
--lambda_cycle=10 \
--lambda_id=5 \
--lambda_ms=0.1 \
--lambda_warp_inv=5 \
--lambda_disp_warp_inv=5 \
2>&1 | tee ./logs/train-$model_name-$now.log &
