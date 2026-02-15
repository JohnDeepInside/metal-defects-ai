"""
Model evaluation — confusion matrix, classification report, model comparison.

Usage:
    python evaluate.py --model lightgbm --data-dir data
    python evaluate.py --model all --data-dir data
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

import config
from dataset import load_and_clean, prepare_splits
from model import ALL_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument("--model", type=str, default="all",
                        choices=ALL_MODELS + ["all"])
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    return parser.parse_args()


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
    class_names: list[str], save_path: Path, title: str = "Confusion Matrix",
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {save_path}")


def plot_per_class_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray,
    class_names: list[str], save_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette("viridis", len(class_names))
    bars = ax.bar(class_names, per_class_acc, color=colors, edgecolor="black")

    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Per-class accuracy saved → {save_path}")


def plot_model_comparison(results_path: Path, save_path: Path) -> None:
    """Bar chart comparing all trained models by val accuracy and F1."""
    with open(results_path) as f:
        data = json.load(f)

    models_data = data["models"]
    names = [m["name"] for m in models_data]
    val_accs = [m["val_acc"] for m in models_data]
    val_f1s = [m["val_f1"] for m in models_data]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width / 2, val_accs, width, label="Accuracy",
                   color="#3498db", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, val_f1s, width, label="F1 Score",
                   color="#e74c3c", edgecolor="black", linewidth=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Validation Set", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Model comparison saved → {save_path}")


def plot_feature_importance(
    clf, feature_names: list[str], save_path: Path, top_n: int = 15,
) -> None:
    """Plot feature importance for tree-based models."""
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        return

    indices = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("viridis", len(top_features))
    ax.barh(top_features, top_importances, color=colors, edgecolor="black",
            linewidth=0.5)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top-{top_n} Feature Importance", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Feature importance saved → {save_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    csv_path = args.data_dir / "train.csv"
    df = load_and_clean(csv_path)
    splits = prepare_splits(df, seed=config.SEED)
    X_test, y_test = splits["test"]
    feature_names = splits["feature_names"]

    model_names = ALL_MODELS if args.model == "all" else [args.model]

    for name in model_names:
        model_path = args.model_dir / f"{name}.joblib"
        if not model_path.exists():
            print(f"Skipping {name} — model file not found: {model_path}")
            continue

        clf = joblib.load(model_path)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"\n{'=' * 50}")
        print(f"Model: {name} | Test Acc: {acc:.4f} | Test F1: {f1:.4f}")
        print(f"{'=' * 50}")

        report = classification_report(
            y_test, y_pred, target_names=config.CLASS_NAMES, digits=4,
        )
        print(report)
        with open(args.output_dir / f"{name}_classification_report.txt", "w") as f:
            f.write(report)

        plot_confusion_matrix(
            y_test, y_pred, config.CLASS_NAMES,
            args.output_dir / f"{name}_confusion_matrix.png",
            title=f"Confusion Matrix — {name}",
        )

        plot_per_class_accuracy(
            y_test, y_pred, config.CLASS_NAMES,
            args.output_dir / f"{name}_per_class_accuracy.png",
        )

        plot_feature_importance(
            clf, feature_names,
            args.output_dir / f"{name}_feature_importance.png",
        )

    # Model comparison plot
    results_path = args.output_dir / "training_results.json"
    if results_path.exists():
        plot_model_comparison(results_path, args.output_dir / "model_comparison.png")


if __name__ == "__main__":
    main()
