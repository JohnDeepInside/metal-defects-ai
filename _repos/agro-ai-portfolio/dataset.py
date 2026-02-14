"""
Data loading utilities for the PlantVillage dataset.

Supports:
  - Train/val transforms with agricultural-image-specific augmentation
  - Standard DataLoaders with shuffling
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config


# ============================================================
# Transforms
# ============================================================
def get_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


# ============================================================
# DataLoaders
# ============================================================
def get_dataloaders(
    data_dir: Path,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> dict[str, DataLoader]:
    """
    Return {train, val} DataLoaders from a PlantVillage-style directory.
    """
    data_dir = Path(data_dir)
    loaders: dict[str, DataLoader] = {}

    for split, tfm, shuffle in [
        ("train", get_train_transforms(), True),
        ("val", get_val_transforms(), False),
    ]:
        split_path = data_dir / split
        if not split_path.exists():
            continue

        ds = datasets.ImageFolder(split_path, transform=tfm)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders
