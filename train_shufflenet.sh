#!/usr/bin/env bash


python main.py \
    --checkpoint_dir=teacher_model_v2 \
    --log_dir=logs_v2 \
    --sample_dir=results_v2 \
    --gpus=0 \
    --batch_size=1 \
    --ch=32 \
    --iteration=25000 \
    --model_name=v2 \
    --dataset=generate_teacher_data \
    --result_dir=results_v2