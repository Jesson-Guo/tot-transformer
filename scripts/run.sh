#!/bin/bash
python main.py \
    --config config/cifar100.yaml \
    --mode train \
    --gpu_ids 0 \
    --backbone resources/weights/ \
    --clip_root resources/weights/ \
    --clip_name Vit-B/32
