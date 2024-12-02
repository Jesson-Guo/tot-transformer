import os
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from timm.data import Mixup
import logging
import argparse

from src.model.image_tot import ImageToT
from src.loss import ToTLoss
from src.dataloader import build_dataloader
from src.optimizer import build_optimizer
from src.scheduler import build_scheduler
from src.config import load_config
from train import Trainer
from eval import Evaluator
from utils import meronyms_with_definition


def setup_logger(log_dir, is_main_process):
    """
    Sets up the logger to log info both to console and a file.
    Only the main process should log to the file in multi-GPU mode.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers in multi-process scenarios
    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler()
        if is_main_process:
            f_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
            f_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            f_handler.setFormatter(formatter)
            logger.addHandler(f_handler)

        c_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        logger.addHandler(c_handler)

    return logger


def main_worker_single(config, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup logger
    logger = setup_logger(config.LOG_DIR, is_main_process=True)
    logger.info(f"Starting {'training' if args.mode == 'train' else 'evaluation'} on single GPU.")

    # Initialize Data Loaders
    if args.mode == 'train':
        train_loader, val_loader = build_dataloader(config, distributed=False)
    elif args.mode == 'eval':
        _, test_loader, _ = build_dataloader(config, distributed=False)
    else:
        raise ValueError(f"Unsupported mode {args.mode}")

    # Initialize Models
    model = ImageToT(
        config,
        num_queries=train_loader.dataset.max_num_mero,
        meronyms=meronyms_with_definition(train_loader.dataset.mero_label_to_idx),
    )
    backbone_state_dict = torch.load(config.MODEL.BACKBONE_ROOT, map_location='cpu')
    model.backbone.load_pretrained(backbone_state_dict['model'])
    model = model.to(device)

    # Initialize Loss Function
    loss_fn = ToTLoss(config, len(train_loader.dataset.mero_labels)).to(device)

    # Initialize Optimizer and Scheduler
    if args.mode == 'train':
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer, len(train_loader))
        # Initialize Mixup
        mixup_fn = None
        mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
                label_smoothing=config.LABEL_SMOOTHING, num_classes=config.DATASET.NUM_CLASSES)

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            mixup_fn=mixup_fn,
            device=device,
            config=config,
            logger=logger
        )

    # Initialize Evaluator
    if args.mode == 'eval':
        evaluator = Evaluator(
            model=model,
            loss_fn=loss_fn,
            device=device,
            config=config,
            logger=logger
        )

    # Execute based on mode
    if args.mode == 'train':
        start_epoch = config.TRAIN.START_EPOCH
        # Resume from checkpoint if specified
        if args.resume:
            if os.path.isfile(args.resume):
                logger.info(f"Loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                logger.info(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            else:
                logger.error(f"No checkpoint found at '{args.resume}'")
                raise FileNotFoundError(f"No checkpoint found at '{args.resume}'")
        else:
            logger.info("No checkpoint provided, starting training from scratch.")

        # Start Training
        trainer.train(train_loader, val_loader, start_epoch=start_epoch)
    elif args.mode == 'eval':
        # Load the best model
        if os.path.isfile(config.MODEL.MODEL_PATH):
            model.load_state_dict(torch.load(config.MODEL.MODEL_PATH, map_location=device))
            logger.info(f"Loaded model weights from {config.MODEL.MODEL_PATH}")
        else:
            logger.error(f"No model found at '{config.MODEL.MODEL_PATH}'")
            raise FileNotFoundError(f"No model found at '{config.MODEL.MODEL_PATH}'")

        # Start Evaluation
        accuracy = evaluator.evaluate(test_loader)
        logger.info(f"Test Accuracy: {accuracy:.2f}%")
    else:
        raise ValueError(f"Unsupported mode {args.mode}")


def main_worker_distributed(local_rank, config, args):
    # Initialize distributed training
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    # Determine if this is the main process
    is_main_process = dist.get_rank() == 0

    # Setup logger
    logger = setup_logger(config.LOG_DIR, is_main_process)
    if is_main_process:
        logger.info(f"Starting {'training' if args.mode == 'train' else 'evaluation'} on Distributed Data Parallel.")

    # Initialize Data Loaders
    if args.mode == 'train':
        train_loader, val_loader = build_dataloader(config, distributed=False)
    elif args.mode == 'eval':
        _, test_loader = build_dataloader(config, distributed=False)
    else:
        raise ValueError(f"Unsupported mode {args.mode}")

    # Initialize Models
    model = ImageToT(
        config,
        num_queries=train_loader.dataset.max_num_mero,
        meronyms=meronyms_with_definition(train_loader.dataset.mero_label_to_idx),
    )
    backbone_state_dict = torch.load(config.MODEL.BACKBONE_ROOT, map_location='cpu')
    model.backbone.load_state_dict(backbone_state_dict)
    model = model.to(device)

    # Wrap model with DDP
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Initialize Loss Function
    loss_fn = ToTLoss(config, len(train_loader.dataset.mero_labels)).to(device)

    # Initialize Optimizer and Scheduler
    if args.mode == 'train':
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer, len(train_loader))
        # Initialize Mixup
        mixup_fn = None
        mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
                label_smoothing=config.LABEL_SMOOTHING, num_classes=config.DATASET.NUM_CLASSES)

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            mixup_fn=mixup_fn,
            device=device,
            config=config,
            logger=logger
        )

    # Initialize Evaluator
    if args.mode == 'eval':
        evaluator = Evaluator(
            model=model,
            loss_fn=loss_fn,
            device=device,
            config=config,
            logger=logger
        )

    # Execute based on mode
    if args.mode == 'train':
        start_epoch = config.TRAIN.START_EPOCH
        # Resume from checkpoint if specified
        if args.resume:
            if os.path.isfile(args.resume):
                if is_main_process:
                    logger.info(f"Loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                if is_main_process:
                    logger.info(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            else:
                if is_main_process:
                    logger.error(f"No checkpoint found at '{args.resume}'")
                raise FileNotFoundError(f"No checkpoint found at '{args.resume}'")
        else:
            if is_main_process:
                logger.info("No checkpoint provided, starting training from scratch.")

        # Start Training
        trainer.train(train_loader, val_loader, start_epoch=start_epoch)
    elif args.mode == 'eval':
        # Load the best model
        if os.path.isfile(config.MODEL.MODEL_PATH):
            model.module.load_state_dict(torch.load(config.MODEL.MODEL_PATH, map_location=device))
            if is_main_process:
                logger.info(f"Loaded model weights from {config.MODEL.MODEL_PATH}")
        else:
            if is_main_process:
                logger.error(f"No model found at '{config.MODEL.MODEL_PATH}'")
            raise FileNotFoundError(f"No model found at '{config.MODEL.MODEL_PATH}'")

        # Start Evaluation
        accuracy = evaluator.evaluate(test_loader)
        if is_main_process:
            logger.info(f"Test Accuracy: {accuracy:.2f}%")
    else:
        raise ValueError(f"Unsupported mode {args.mode}")

    # Cleanup
    if torch.distributed.is_initialized():
        dist.destroy_process_group()


def main():
    """
    The main entry point of the script.
    Parses command-line arguments, sets environment variables, and launches training/evaluation.
    """
    parser = argparse.ArgumentParser(description='Multi-Stage Learnable CoT-Transformer Training and Evaluation')
    parser.add_argument('--config', type=str, default='config/cifar100.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help='Mode: train or eval')
    parser.add_argument('--distributed', action='store_true', help='Use multi-GPU distributed training')
    parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated GPU IDs to use (e.g., "0,1,2")')
    parser.add_argument('--resume', type=str, default='', help='Path to resume checkpoint')
    # Add more arguments as needed to override config parameters

    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES based on --gpu_ids
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Load YAML config
    config = load_config(args.config)

    # Decide whether to use distributed training
    if args.distributed and torch.cuda.device_count() > 1:
        # Multi-GPU training using Distributed Data Parallel (DDP)
        world_size = torch.cuda.device_count()
        print(f"Launching distributed training on {world_size} GPUs: {args.gpu_ids}")
        mp.spawn(
            main_worker_distributed,
            nprocs=world_size,
            args=(config, args,)
        )
    else:
        # Single-GPU training or evaluation
        if args.distributed and torch.cuda.device_count() == 1:
            print("Distributed training requested but only one GPU is available. Falling back to single GPU.")
        print("Launching single-GPU training/evaluation.")
        main_worker_single(config, args)


if __name__ == '__main__':
    main()
