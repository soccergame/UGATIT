#!/usr/bin/env bash


python main.py \
    --checkpoint_dir=teacher_model_v2 \
    --log_dir=logs_v2 \
    --sample_dir=results_v2 \
    --gpus=1 \
    --batch_size=2 \
    --ch=64 \
    --light=True \
    --iteration=26000 \
    --model_name=v2 \
    --lr=0.00001 \
    --cam_weight=1 \
    --dataset=generate_teacher_data \
    --result_dir=results_v2