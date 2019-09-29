#!/usr/bin/env bash


python main.py \
    --checkpoint_dir=teacher_model_v3 \
    --log_dir=logs_v3 \
    --sample_dir=results_v3 \
    --gpus=1 \
    --batch_size=1 \
    --ch=64 \
    --light=True \
    --iteration=7000 \
    --model_version=v3 \
    --lr=0.00001 \
    --cam_weight=1 \
    --dataset=generate_teacher_data \
    --result_dir=results_v3