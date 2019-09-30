#!/usr/bin/env bash


python main.py \
    --checkpoint_dir=teacher_model_v1 \
    --log_dir=logs_v1 \
    --sample_dir=results_v1 \
    --gpus=0 \
    --n_downsample=4 \
    --batch_size=1 \
    --img_size=512 \
    --ch=32 \
    --iteration=28000 \
    --dataset=/search/datasets/generate_teacher \
    --result_dir=results_v1