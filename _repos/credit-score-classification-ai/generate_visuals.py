"""
Generate presentation-ready visualizations for the README.

Reads training_results.json and trained models to produce:
  - model_comparison.png
  - confusion_matrix.png (best model)
  - per_class_accuracy.png (best model)
  - feature_importance.png (best model)

Usage:
    python generate_visuals.py
    python generate_visuals.py --data-dir data --results-dir outputs/results
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import config
from dataset import load_and_clean, prepare_splits


ASSETS_DIR = config.PROJECT_ROOT / "assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate README visuals.")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    return parser.parse_args()


def generate_model_comparison(results_path: Path, save_path: Path) -> None:
    with open(results_path) as f:
        data = json.load(f)

    models = data["models"]
    names = [m["name"] for m in models]
    val_accs = [m["val_acc"] for m in models]
    val_f1s = [m["val_f1"] for m in models]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width / 2, val_accs, width, label="Accuracy",
                   color="#3498db", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, val_f1s, width, label="F1 Score",
                   color="#e74c3c", edgecolor="black", linewidth=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Validation Set",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def generate_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
    class_names: list[str], save_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Best Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def generate_per_class_accuracy(
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
    print(f"Saved → {save_path}")


def generate_feature_importance(
    clf, feature_names: list[str], save_path: Path, top_n: int = 15,
) -> None:
    if not hasattr(clf, "feature_importances_"):
        return

    importances = clf.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("viridis", len(top_features))
    ax.barh(top_features, top_importances, color=colors, edgecolor="black",
            linewidth=0.5)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top-{top_n} Feature Importance",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def main() -> None:
    args = parse_args()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    results_path = args.results_dir / "training_results.json"
    with open(results_path) as f:
        results = json.load(f)

    best_model_name = results["best_model"]
    feature_names = results["feature_names"]

    # Load best model
    clf = joblib.load(args.model_dir / f"{best_model_name}.joblib")

    # Load test data
    df = load_and_clean(args.data_dir / "train.csv")
    splits = prepare_splits(df, seed=config.SEED)
    X_test, y_test = splits["test"]
    y_pred = clf.predict(X_test)

    # Generate all plots
    generate_model_comparison(results_path, ASSETS_DIR / "model_comparison.png")
    generate_confusion_matrix(y_test, y_pred, config.CLASS_NAMES,
                              ASSETS_DIR / "confusion_matrix.png")
    generate_per_class_accuracy(y_test, y_pred, config.CLASS_NAMES,
                                ASSETS_DIR / "per_class_accuracy.png")
    generate_feature_importance(clf, feature_names,
                                ASSETS_DIR / "feature_importance.png")

    print(f"\nAll visuals generated in {ASSETS_DIR}/")


if __name__ == "__main__":
    main()
