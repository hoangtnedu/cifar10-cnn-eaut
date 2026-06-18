from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, eval_transform


def get_cifar10_loaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_dir = Path(data_dir)
    train_transform, eval_transform = build_transforms()

    train_full = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=train_transform)
    val_full = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=eval_transform)
    test_set = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=eval_transform)

    n = len(train_full)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = int(n * val_ratio)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()

    train_set = Subset(train_full, train_indices)
    val_set = Subset(val_full, val_indices)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader
