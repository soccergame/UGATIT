#!/usr/bin/env bash


python main.py \
    --checkpoint_dir=teacher_model \
    --phase=test \
    --log_dir=logs \
    --sample_dir=results \
    --gpus=0 \
    --batch_size=1 \
    --dataset=generate_teacher_data \
    --result_dir=results