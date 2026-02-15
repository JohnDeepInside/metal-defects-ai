"""
Training script for Credit Score Classification.

Usage:
    python train.py --data-dir data --model lightgbm
    python train.py --model all
"""

import json
import logging
import random
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import config
from dataset import load_and_clean, prepare_splits
from model import ALL_MODELS, build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def train_single(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> dict:
    """Train a single model and return metrics."""
    logger.info("Training: %s", name)
    t0 = time.time()

    clf = build_model(name, seed=seed)
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    val_f1 = f1_score(y_val, clf.predict(X_val), average="weighted")
    elapsed = time.time() - t0

    logger.info(
        "  %s | Train Acc: %.4f | Val Acc: %.4f | Val F1: %.4f | %.1fs",
        name, train_acc, val_acc, val_f1, elapsed,
    )

    return {
        "model": clf,
        "name": name,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "elapsed": elapsed,
    }


def main() -> None:
    args = config.parse_args()
    set_seed(args.seed)

    model_dir = args.output_dir / "models"
    results_dir = args.output_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    csv_path = args.data_dir / "train.csv"
    logger.info("Loading data from %s", csv_path)
    df = load_and_clean(csv_path)
    splits = prepare_splits(df, test_size=args.test_size, seed=args.seed)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    scaler = splits["scaler"]
    feature_names = splits["feature_names"]

    logger.info(
        "Data splits — train: %d | val: %d | test: %d | features: %d",
        len(y_train), len(y_val), len(y_test), X_train.shape[1],
    )

    # Save scaler
    joblib.dump(scaler, model_dir / "scaler.joblib")
    logger.info("Scaler saved → %s", model_dir / "scaler.joblib")

    # Determine which models to train
    model_names = ALL_MODELS if args.model == "all" else [args.model]

    results = []
    best_val_acc = 0.0
    best_model_name = ""

    for name in model_names:
        result = train_single(name, X_train, y_train, X_val, y_val, args.seed)

        # Save model
        save_path = model_dir / f"{name}.joblib"
        joblib.dump(result["model"], save_path)
        logger.info("  Model saved → %s", save_path)

        if result["val_acc"] > best_val_acc:
            best_val_acc = result["val_acc"]
            best_model_name = name

        results.append({
            "name": result["name"],
            "train_acc": result["train_acc"],
            "val_acc": result["val_acc"],
            "val_f1": result["val_f1"],
            "elapsed": result["elapsed"],
        })

    # Evaluate best model on test set
    logger.info("Best model: %s (val acc: %.4f)", best_model_name, best_val_acc)
    best_clf = joblib.load(model_dir / f"{best_model_name}.joblib")
    test_acc = accuracy_score(y_test, best_clf.predict(X_test))
    test_f1 = f1_score(y_test, best_clf.predict(X_test), average="weighted")
    logger.info("Test set — Acc: %.4f | F1: %.4f", test_acc, test_f1)

    # Save results
    summary = {
        "best_model": best_model_name,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "feature_names": feature_names,
        "class_names": config.CLASS_NAMES,
        "models": results,
    }
    with open(results_dir / "training_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved → %s", results_dir / "training_results.json")


if __name__ == "__main__":
    main()
