import os
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import logging as transformers_logging
from timm.data import Mixup
import logging
from datetime import datetime

from src.model.tot_transformer import MultiStageCoT
from src.dataloader import get_dataloader
from src.loss import MultiStageLoss
from src.optimizer import get_optimizer
from src.scheduler import get_scheduler
from src.mg_graph import MultiGranGraph
from train import Trainer
from eval import Evaluator

# Suppress some warnings from transformers
transformers_logging.set_verbosity_error()

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main_worker(local_rank, config):
    # Setup distributed training
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    # Setup logger
    log_dir = config['log_dir']
    logger = setup_logger(log_dir)
    if local_rank == 0:
        logger.info(f"Starting process with config: {config}")
    
    # Initialize Multi-Granularity Structure Graph
    mg_graph = MultiGranGraph(config['mg_graph'])
    labels_per_stage = mg_graph.labels_per_stage  # List of lists
    num_classes_per_stage = mg_graph.get_num_classes_per_stage()
    
    # Initialize Data Loaders
    if config['mode'] == 'train':
        train_loader, val_loader, _ = get_dataloader(config, mg_graph, distributed=True)
    elif config['mode'] == 'eval':
        test_loader, _, _ = get_dataloader(config, mg_graph, distributed=True)
    else:
        raise ValueError(f"Unsupported mode {config['mode']}")
    
    # Initialize Models
    model = MultiStageCoT(
        num_stages=config['num_stages'],
        num_classes_per_stage=num_classes_per_stage,
        prompt_dim=config['model']['prompt_dim'],
        label_names_per_stage=labels_per_stage,
        clip_model_name=config['model']['clip_model_name'],
        pretrained=config['model']['pretrained']
    ).to(device)
    
    # Wrap model with DDP
    if config['mode'] == 'train':
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    
    # Initialize Loss Function
    if config['mode'] == 'train':
        loss_fn = MultiStageLoss(
            num_stages=config['num_stages'],
            mg_graph=mg_graph,
            alpha=config['loss']['alpha'],
            beta=config['loss']['beta'],
            gamma=config['loss']['gamma'],
            lambda_eval=config['loss']['lambda_eval']
        ).to(device)
    
    # Initialize Optimizer and Scheduler
    if config['mode'] == 'train':
        optimizer = get_optimizer(model, config['optimizer'])
        scheduler = get_scheduler(optimizer, config['scheduler'])
    
        # Initialize Mixup
        mixup_fn = Mixup(
            mixup_alpha=config['mixup']['alpha'],
            cutmix_alpha=config['mixup']['cutmix_alpha'],
            label_smoothing=config['mixup']['label_smoothing'],
            num_classes=num_classes_per_stage[-1]  # Last stage has fine-grained labels
        )
    
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
    if config['mode'] == 'eval':
        # Load the best model
        if local_rank == 0:
            model_path = config['model']['model_path']
            model.module.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Loaded model weights from {model_path}")
        dist.barrier()  # Ensure all processes have loaded the model
    
        # Initialize loss function if needed
        loss_fn = MultiStageLoss(
            num_stages=config['num_stages'],
            mg_graph=mg_graph,
            alpha=config['loss']['alpha'],
            beta=config['loss']['beta'],
            gamma=config['loss']['gamma'],
            lambda_eval=config['loss']['lambda_eval']
        ).to(device)
    
        evaluator = Evaluator(
            model=model,
            loss_fn=loss_fn,
            device=device,
            config=config,
            logger=logger
        )
    
    # Execute based on mode
    if config['mode'] == 'train':
        # Start Training
        trainer.train(train_loader, val_loader)
    
        # Save Models (only save from rank 0)
        if local_rank == 0:
            torch.save(model.module.state_dict(), os.path.join(log_dir, 'multi_stage_cot.pt'))
            logger.info("Models saved successfully.")
    elif config['mode'] == 'eval':
        # Start Evaluation
        accuracy = evaluator.evaluate(test_loader)
        if local_rank == 0:
            logger.info(f"Test Accuracy: {accuracy:.2f}%")
    else:
        raise ValueError(f"Unsupported mode {config['mode']}")
    
    dist.destroy_process_group()

def main():
    # Assume that the config path is provided as an environment variable or argument
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Stage Learnable CoT-Transformer Training and Evaluation')
    parser.add_argument('--config', type=str, default='src/config/cifar100.yaml', help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)

    # Set up distributed training launch
    world_size = torch.cuda.device_count()
    mp.spawn(
        main_worker,
        nprocs=world_size,
        args=(config,)
    )

if __name__ == '__main__':
    main()
