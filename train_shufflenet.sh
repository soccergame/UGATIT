#!/usr/bin/env bash


python main.py \
    --checkpoint_dir=teacher_model_v1 \
    --log_dir=logs \
    --sample_dir=results \
    --gpus=0 \
    --batch_size=1 \
    --ch=16 \
    --dataset=generate_teacher_data \
    --result_dir=results