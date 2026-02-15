"""
Centralized configuration for the Credit Score Classification project.
"""

import argparse
from pathlib import Path


# ============================================================
# Default paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

# ============================================================
# Dataset
# ============================================================
CLASS_NAMES = ["Poor", "Standard", "Good"]
NUM_CLASSES = len(CLASS_NAMES)

LABEL_MAP = {"Poor": 0, "Standard": 1, "Good": 2}

# Features used for training (after cleaning)
NUMERIC_FEATURES = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio",
    "Total_EMI_per_month", "Amount_invested_monthly",
    "Monthly_Balance",
]

CATEGORICAL_FEATURES = [
    "Occupation", "Credit_Mix", "Payment_of_Min_Amount",
    "Payment_Behaviour",
]

# ============================================================
# Training defaults
# ============================================================
TEST_SIZE = 0.2
VAL_SIZE = 0.1
SEED = 42
N_JOBS = -1

# ============================================================
# Model hyperparameters
# ============================================================
LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "multi_class": "multinomial",
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "learning_rate": 0.1,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
}

CATBOOST_PARAMS = {
    "iterations": 300,
    "depth": 8,
    "learning_rate": 0.1,
    "verbose": 0,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with sensible defaults."""
    parser = argparse.ArgumentParser(
        description="Train / evaluate a credit-score classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory containing train.csv / test.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Directory for models and results")

    # Training
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)

    # Model
    parser.add_argument("--model", type=str, default="lightgbm",
                        choices=["logistic_regression", "random_forest",
                                 "xgboost", "lightgbm", "catboost", "all"],
                        help="Model to train (or 'all' to train every model)")

    return parser.parse_args()
