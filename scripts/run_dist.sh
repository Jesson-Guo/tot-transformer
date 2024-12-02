#!/bin/bash
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12347 \
    main.py \
    --config config/cifar100.yaml \
    --mode train \
    --gpu_ids 0,1,2,3 \
    --distributed
