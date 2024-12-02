# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from .dataset.cub import CUB
from .dataset.mero import MeroDataset


try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms


    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_dataloader(config, distributed=False):
    dataset_train = build_dataset(is_train=True, config=config)
    if distributed:
        print(f"local rank {dist.get_rank()} / global rank {dist.get_rank()} successfully build train dataset")
    else:
        print("successfully build train dataset")

    dataset_val = build_dataset(is_train=False, config=config)
    if distributed:
        print(f"local rank {dist.get_rank()} / global rank {dist.get_rank()} successfully build val dataset")
    else:
        print("successfully build val dataset")

    if distributed:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = RandomSampler(dataset_train)

    if distributed:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=False
        )
    else:
        sampler_val = SequentialSampler(dataset_val)

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.DATASET.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.DATASET.PIN_MEMORY,
        drop_last=False
    )

    return data_loader_train, data_loader_val


def build_dataset(is_train, config):
    if config.DATASET.NAME == 'imagenet':
        prefix = 'train' if is_train else 'val'
        dataset = datasets.ImageFolder(
            root=os.path.join(config.DATASET.DATA_PATH, prefix),
            transform=build_transform(is_train, config)
        )
    elif config.DATASET.NAME == 'cifar100':
        dataset = datasets.CIFAR100(
            root=config.DATASET.DATA_PATH,
            train=is_train,
            transform=build_transform(is_train, config)
        )
    elif config.DATASET.NAME == 'stanford_cars':
        split = 'train' if is_train else 'test'
        dataset = datasets.StanfordCars(
            root=config.DATASET.DATA_PATH,
            split=split,
            transform=build_transform(is_train, config)
        )
    elif config.DATASET.NAME == 'cub':
        split = 'train' if is_train else 'test'
        dataset = CUB(
            root=config.DATASET.DATA_PATH,
            split=split,
            transform=build_transform(is_train, config)
        )
    else:
        raise NotImplementedError(f"Dataset {config.DATASET.NAME} not supported.")

    dataset = MeroDataset(config.DATASET.NUM_CLASSES, dataset, config.DATASET.HIERARCHY)

    return dataset


def build_transform(is_train, config):
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATASET.IMAGE_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATASET.INTERPOLATION,
        )
        return transform

    t = []
    # test crop
    size = int((256 / 224) * config.DATASET.IMAGE_SIZE)
    t.append(
        transforms.Resize(size, interpolation=_pil_interp(config.DATASET.INTERPOLATION)),
        # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(config.DATASET.IMAGE_SIZE))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
