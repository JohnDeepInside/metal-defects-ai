"""
Data loading utilities: transforms, train/val/test split, weighted sampling.
"""

import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
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
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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
# Dataset split
# ============================================================
def split_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    seed: int = config.SEED,
) -> None:
    """
    Split a flat ImageFolder dataset into train / val / test folders.

    Expected input layout:
        source_dir/
            ClassName1/
                img001.jpg
                ...
            ClassName2/
                ...

    Resulting layout:
        output_dir/
            train/ val/ test/  — each with class sub-folders.
    """
    random.seed(seed)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    for split in ("train", "val", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        images = sorted(class_dir.glob("*"))
        images = [p for p in images if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train: n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, file_list in splits.items():
            dst = output_dir / split_name / class_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for fp in file_list:
                shutil.copy2(fp, dst / fp.name)

    print(f"Dataset split complete → {output_dir}")
    for split in ("train", "val", "test"):
        split_path = output_dir / split
        total = sum(1 for _ in split_path.rglob("*") if _.is_file())
        print(f"  {split}: {total} images")


# ============================================================
# DataLoaders
# ============================================================
def get_dataloaders(
    data_dir: Path,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> dict[str, DataLoader]:
    """
    Return {train, val, test} DataLoaders from a pre-split directory.
    Training loader uses WeightedRandomSampler for class balance.
    """
    data_dir = Path(data_dir)
    loaders: dict[str, DataLoader] = {}

    for split, tfm in [
        ("train", get_train_transforms()),
        ("val", get_val_transforms()),
        ("test", get_val_transforms()),
    ]:
        split_path = data_dir / split
        if not split_path.exists():
            continue

        ds = datasets.ImageFolder(split_path, transform=tfm)

        sampler = None
        shuffle = False
        if split == "train":
            class_counts = np.array(
                [len(list((split_path / c).iterdir())) for c in ds.classes]
            )
            weights = 1.0 / class_counts
            sample_weights = [weights[label] for _, label in ds.samples]
            sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True,
            )

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders
