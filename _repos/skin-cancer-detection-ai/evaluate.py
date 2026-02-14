"""
Model evaluation — confusion matrix, classification report, training curves.

Usage:
    python evaluate.py --checkpoint outputs/checkpoints/best_model.pth --data-dir data
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

import config
from dataset import get_val_transforms
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--split", type=str, default="val", choices=["val"])
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=config.RESULTS_DIR)
    return parser.parse_args()


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        all_labels.append(labels.numpy())
        all_preds.append(outputs.argmax(1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
    class_names: list[str], save_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {save_path}")


def plot_training_curves(history_path: Path, save_path: Path) -> None:
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "o-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "o-", label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved → {save_path}")


def plot_per_class_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray,
    class_names: list[str], save_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("viridis", len(class_names))
    bars = ax.bar(class_names, per_class_acc, color=colors, edgecolor="black")

    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Per-class accuracy saved → {save_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    arch = ckpt.get("arch", "mobilenet_v2")
    num_classes = ckpt.get("num_classes", config.NUM_CLASSES)
    class_names = ckpt.get("class_names", config.CLASS_NAMES)

    model = build_model(arch=arch, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    split_path = args.data_dir / args.split
    ds = datasets.ImageFolder(split_path, transform=get_val_transforms())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Evaluating on {args.split} set: {len(ds)} images")

    y_true, y_pred, _ = collect_predictions(model, loader, device)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n" + report)
    with open(args.output_dir / "classification_report.txt", "w") as f:
        f.write(report)

    plot_confusion_matrix(y_true, y_pred, class_names,
                          args.output_dir / "confusion_matrix.png")
    plot_per_class_accuracy(y_true, y_pred, class_names,
                            args.output_dir / "per_class_accuracy.png")

    history_path = args.output_dir / "history.json"
    if history_path.exists():
        plot_training_curves(history_path, args.output_dir / "training_curves.png")


if __name__ == "__main__":
    main()
