#!/bin/bash
CUDA_VISIBLE_DEVICES=5,1,4,3 torchrun --nproc_per_node=4 --master_port 12347 \
    main.py \
    --config config/cifar100.yaml \
    --mode train \
    --distributed
