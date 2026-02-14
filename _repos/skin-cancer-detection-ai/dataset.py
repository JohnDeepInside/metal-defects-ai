"""
Data loading utilities for the HAM10000 skin lesion dataset.

Supports:
  - Organizing raw images from CSV + image directories into class folders
  - Train/val transforms with medical-image-specific augmentation
  - Weighted sampling for severe class imbalance
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
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
# Organize dataset from CSV
# ============================================================
def organize_dataset(
    csv_path: str | Path,
    image_dirs: list[str | Path],
    output_dir: str | Path,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> None:
    """
    Organize HAM10000 images into train/val folders by class.

    Args:
        csv_path: Path to HAM10000_metadata.csv
        image_dirs: List of directories containing images (part_1, part_2)
        output_dir: Where to create train/ and val/ folders
        test_size: Fraction of data for validation
        seed: Random seed for reproducibility
    """
    output_dir = Path(output_dir)
    df = pd.read_csv(csv_path)
    print(f"Total images: {len(df)}")
    print(f"Class distribution:\n{df['dx'].value_counts()}\n")

    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["dx"],
    )

    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        for _, row in split_df.iterrows():
            filename = row["image_id"] + ".jpg"

            src = None
            for img_dir in image_dirs:
                candidate = Path(img_dir) / filename
                if candidate.exists():
                    src = candidate
                    break

            if src is None:
                continue

            dst_folder = output_dir / split_name / row["dx"]
            dst_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_folder / filename)

    print(f"Dataset organized → {output_dir}")
    for split in ("train", "val"):
        total = sum(1 for _ in (output_dir / split).rglob("*") if _.is_file())
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
    Return {train, val} DataLoaders from an organized directory.
    Training loader uses WeightedRandomSampler for class balance.
    """
    data_dir = Path(data_dir)
    loaders: dict[str, DataLoader] = {}

    for split, tfm in [
        ("train", get_train_transforms()),
        ("val", get_val_transforms()),
    ]:
        split_path = data_dir / split
        if not split_path.exists():
            continue

        ds = datasets.ImageFolder(split_path, transform=tfm)

        sampler = None
        shuffle = False
        if split == "train":
            class_counts = np.array(
                [len(os.listdir(split_path / c)) for c in ds.classes]
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
