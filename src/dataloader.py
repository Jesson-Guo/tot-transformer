import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR100, ImageFolder
from timm.data import Mixup
import os

from src.mg_graph import MultiGranGraph

def get_dataloader(config, mg_graph, distributed=True):
    """
    Returns train and validation dataloaders.
    """
    dataset_name = config['dataset']['name']
    data_dir = config['dataset']['data_dir']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    
    # Define transformations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(config['dataset']['image_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config['dataset']['mean'], config['dataset']['std'])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(config['dataset']['image_size']),
        transforms.CenterCrop(config['dataset']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(config['dataset']['mean'], config['dataset']['std'])
    ])
    
    # Load Dataset
    if dataset_name.lower() == 'cifar100':
        train_dataset = CIFAR100(root=data_dir, train=True, download=True, transform=train_transforms)
        val_dataset = CIFAR100(root=data_dir, train=False, download=True, transform=val_transforms)
    elif dataset_name.lower() == 'imagenet':
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
        val_dataset = ImageFolder(root=val_dir, transform=val_transforms)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")
    
    # Wrap datasets to include multi-granularity labels
    train_dataset = MultiGranularityDataset(train_dataset, mg_graph, config['num_stages'])
    val_dataset = MultiGranularityDataset(val_dataset, mg_graph, config['num_stages'])
    
    # Samplers
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Number of classes per stage
    num_classes_per_stage = mg_graph.get_num_classes_per_stage()
    
    return train_loader, val_loader, num_classes_per_stage

class MultiGranularityDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, mg_graph, num_stages):
        self.base_dataset = base_dataset
        self.mg_graph = mg_graph
        self.num_stages = num_stages
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, fine_label = self.base_dataset[idx]
        label_sequence = self.mg_graph.get_label_sequence(fine_label)
        # Ensure label_sequence has length equal to num_stages
        if len(label_sequence) < self.num_stages:
            # Pad with root labels or any appropriate strategy
            label_sequence = [label_sequence[0]] * (self.num_stages - len(label_sequence)) + label_sequence
        elif len(label_sequence) > self.num_stages:
            label_sequence = label_sequence[-self.num_stages:]
    
        # Convert labels to indices per stage
        label_indices = []
        for t in range(self.num_stages):
            label = label_sequence[t]
            index = self.mg_graph.label_to_index[t].get(label, 0)  # Default to 0 if label not found
            label_indices.append(index)
    
        return image, label_indices
