"""
Training script for Skin Cancer Detection.

Usage:
    python train.py --data-dir data --epochs 20 --model mobilenet_v2
    python train.py --resume outputs/checkpoints/best_model.pth
"""

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from dataset import get_dataloaders
from model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main() -> None:
    args = config.parse_args()
    set_seed(args.seed)

    checkpoint_dir = args.output_dir / "checkpoints"
    results_dir = args.output_dir / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    loaders = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    logger.info(
        "Dataset loaded — train: %d | val: %d",
        len(loaders["train"].dataset),
        len(loaders["val"].dataset),
    )
    logger.info("Classes: %s", loaders["train"].dataset.classes)

    model = build_model(arch=args.model, pretrained=args.pretrained).to(device)
    logger.info("Model: %s (%s params)", args.model,
                f"{sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=config.SCHEDULER_PATIENCE,
        factor=config.SCHEDULER_FACTOR,
    )

    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        logger.info("Resumed from epoch %d (best val acc: %.4f)", start_epoch, best_val_acc)

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "lr": [],
    }
    epochs_no_improve = 0

    logger.info("Starting training for %d epochs...", args.epochs)
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device,
        )
        val_loss, val_acc = validate(model, loaders["val"], criterion, device)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        logger.info(
            "Epoch %2d/%d | Train Loss: %.4f  Acc: %.4f | "
            "Val Loss: %.4f  Acc: %.4f | LR: %.1e",
            epoch + 1, args.epochs,
            train_loss, train_acc, val_loss, val_acc, lr,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "arch": args.model,
                    "num_classes": config.NUM_CLASSES,
                    "class_names": config.CLASS_NAMES,
                },
                checkpoint_dir / "best_model.pth",
            )
            logger.info("  ✓ Best model saved (val acc: %.4f)", best_val_acc)
        else:
            epochs_no_improve += 1

        if args.patience > 0 and epochs_no_improve >= args.patience:
            logger.info("Early stopping after %d epochs without improvement.", args.patience)
            break

    elapsed = time.time() - t_start
    logger.info("Training complete in %.1f s — Best Val Acc: %.4f", elapsed, best_val_acc)

    with open(results_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved → %s", results_dir / "history.json")


if __name__ == "__main__":
    main()
