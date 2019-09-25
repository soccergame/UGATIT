#!/usr/bin/env bash


python main.py \
    --checkpoint_dir=teacher_model_v1 \
    --log_dir=logs_v1 \
    --sample_dir=results_v1 \
    --gpus=0 \
    --batch_size=4 \
    --ch=32 \
    --iteration=26000 \
    --dataset=generate_teacher_data \
    --result_dir=results_v1