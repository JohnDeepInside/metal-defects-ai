"""
Single-sample inference for Credit Score Classification.

Usage:
    python inference.py --model-dir outputs/models --model lightgbm
    python inference.py --model-dir outputs/models --model lightgbm --csv new_data.csv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
from dataset import load_and_clean, get_feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict credit scores.")
    parser.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    parser.add_argument("--model", type=str, default="lightgbm",
                        choices=["logistic_regression", "random_forest",
                                 "xgboost", "lightgbm", "catboost"])
    parser.add_argument("--csv", type=Path, default=None,
                        help="CSV file with samples to classify")
    return parser.parse_args()


def predict(
    model_dir: Path,
    model_name: str,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a trained model + scaler and predict.

    Returns:
        (predicted_classes, probabilities)
    """
    scaler = joblib.load(model_dir / "scaler.joblib")
    clf = joblib.load(model_dir / f"{model_name}.joblib")

    X_scaled = scaler.transform(X)
    predictions = clf.predict(X_scaled)

    if hasattr(clf, "predict_proba"):
        probabilities = clf.predict_proba(X_scaled)
    else:
        probabilities = np.zeros((len(predictions), config.NUM_CLASSES))

    return predictions, probabilities


def main() -> None:
    args = parse_args()

    model_path = args.model_dir / f"{args.model}.joblib"
    scaler_path = args.model_dir / "scaler.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    if args.csv and args.csv.exists():
        df = load_and_clean(args.csv)
        feature_cols = [c for c in get_feature_columns() if c in df.columns]
        X = df[feature_cols].values.astype(np.float32)

        predictions, probabilities = predict(args.model_dir, args.model, X)
        class_labels = [config.CLASS_NAMES[p] for p in predictions]

        print(f"\nModel: {args.model}")
        print(f"Samples: {len(predictions)}")
        print(f"\nPrediction distribution:")
        for cls in config.CLASS_NAMES:
            count = class_labels.count(cls)
            print(f"  {cls:10s}: {count:5d} ({count / len(class_labels):.1%})")

        # Save predictions
        result_df = pd.DataFrame({
            "prediction": class_labels,
            "confidence": [probabilities[i, p] for i, p in enumerate(predictions)],
        })
        save_path = args.model_dir.parent / "results" / "predictions.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(save_path, index=False)
        print(f"\nPredictions saved → {save_path}")
    else:
        print("Interactive mode — provide features manually or pass --csv")
        print(f"Expected features: {get_feature_columns()}")


if __name__ == "__main__":
    main()
