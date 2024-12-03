import os
import sys
import json
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import logging
import argparse
import functools

from timm.data import Mixup
from termcolor import colored
from src.model.image_tot import ImageToT
from src.loss import ToTLoss
from src.dataloader import build_dataloader
from src.optimizer import build_optimizer
from src.scheduler import build_scheduler
from src.config import load_config
from train import Trainer
from utils import NativeScalerWithGradNormCount, meronyms_with_definition, is_main_process


@functools.lru_cache()
def setup_logger(log_dir, is_main_process):
    """
    Sets up the logger to log info both to console and a file.
    Only the main process should log to the file in multi-GPU mode.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    if is_main_process == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'), mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def print_configs(args, config, model, logger):
    if is_main_process():
        logger.info(f"Starting {'training' if args.mode == 'train' else 'evaluation'} on Distributed Data Parallel.")
        logger.info(config.dump())
        logger.info(json.dumps(vars(args)))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        if is_main_process():
            logger.info(f"number of GFLOPs: {flops / 1e9}")


def parse_option():
    parser = argparse.ArgumentParser(description='Multi-Stage Learnable CoT-Transformer Training and Evaluation')
    parser.add_argument('--config', type=str, default='config/cifar100.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='eval', help='Mode: train or eval')
    parser.add_argument('--distributed', action='store_true', help='Use multi-GPU distributed training')
    parser.add_argument('--resume', type=str, default='', help='Path to resume checkpoint')

    args = parser.parse_args()

    config = load_config(args.config)

    return args, config
    

def main():
    args, config = parse_option()
    if not args.distributed:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        world_size = torch.cuda.device_count()
        if is_main_process():
            print(f"Launching distributed training on {world_size} GPUs")

        # Initialize distributed training
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    else:
        # Single-GPU training or evaluation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if is_main_process():
            print("Launching single-GPU.")

    # Setup logger
    logger = setup_logger(config.LOG_DIR, is_main_process())
    os.makedirs(os.path.join(config.LOG_DIR, config.DATASET.NAME), exist_ok=True)

    # Initialize Data Loaders
    train_loader, val_loader = build_dataloader(config, distributed=args.distributed)

    # Initialize Models
    model = ImageToT(
        config,
        num_queries=val_loader.dataset.max_num_mero,
        meronyms=meronyms_with_definition(val_loader.dataset.mero_label_to_idx),
    ).to(device)

    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, len(val_loader))

    print_configs(args, config, model, logger)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    # Initialize Loss Function
    criterion = ToTLoss(config, len(val_loader.dataset.mero_labels)).to(device)

    # Load backbone weights
    checkpoint = torch.load(config.MODEL.BACKBONE_ROOT, map_location='cpu')
    backbone_state_dict = checkpoint["model"]
    model_without_ddp.base_class_head.load_state_dict({'weight': backbone_state_dict['head.weight'], 'bias': backbone_state_dict['head.bias']})
    del backbone_state_dict['head.weight'], backbone_state_dict['head.bias']
    model_without_ddp.backbone.load_state_dict(backbone_state_dict)
    if is_main_process():
        logger.info("Pretrained weights loaded successfully into Backbone.")

    # Resume from checkpoint if specified
    if args.mode == 'train':
        start_epoch = config.TRAIN.START_EPOCH
        # Resume from checkpoint if specified
        if args.resume:
            if os.path.isfile(args.resume):
                if is_main_process():
                    logger.info(f"Loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume, map_location=device)
                model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                if is_main_process():
                    logger.info(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            else:
                if is_main_process():
                    logger.error(f"No checkpoint found at '{args.resume}'")
                raise FileNotFoundError(f"No checkpoint found at '{args.resume}'")
    elif args.mode == 'eval':
        # Load the best model
        if os.path.isfile(config.MODEL.MODEL_PATH):
            model_without_ddp.load_state_dict(torch.load(config.MODEL.MODEL_PATH, map_location=device))
            if is_main_process():
                logger.info(f"Loaded model weights from {config.MODEL.MODEL_PATH}")
        else:
            if is_main_process():
                logger.info(f"No model loaded")
    else:
        raise ValueError(f"Unsupported mode {args.mode}")

    # Initialize Mixup
    mixup_fn = None
    # mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    # if mixup_active:
    #     mixup_fn = Mixup(
    #         mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
    #         prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
    #         label_smoothing=config.LABEL_SMOOTHING, num_classes=config.DATASET.NUM_CLASSES)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        model_without_ddp=model_without_ddp,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        mixup_fn=mixup_fn,
        device=device,
        config=config,
        logger=logger,
        scaler=NativeScalerWithGradNormCount()
    )

    if args.mode == 'train':
        trainer.train(train_loader, val_loader, start_epoch=start_epoch)
    elif args.mode == 'eval':
        accuracy = trainer.validate(val_loader)
        if is_main_process():
            logger.info(f"Test Accuracy: {accuracy:.2f}%")
    else:
        raise ValueError(f"Unsupported mode {args.mode}")

    # Cleanup
    if torch.distributed.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    seed = 2
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    main()
