"""
Data loading and preprocessing for the Credit Score Classification dataset.

Supports:
  - Loading raw CSV from Kaggle
  - Cleaning noisy values (special characters, missing data)
  - Feature engineering from raw columns
  - Train/val/test split with stratification
  - Feature scaling with StandardScaler
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config


# ============================================================
# Cleaning helpers
# ============================================================
def _clean_numeric(series: pd.Series) -> pd.Series:
    """Strip non-numeric characters and convert to float."""
    return (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )


def _parse_credit_history_months(series: pd.Series) -> pd.Series:
    """Convert '22 Years and 1 Months' → total months as float."""
    def _to_months(val):
        if pd.isna(val):
            return np.nan
        match = re.match(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?", str(val))
        if match:
            return int(match.group(1)) * 12 + int(match.group(2))
        return np.nan
    return series.apply(_to_months)


# ============================================================
# Main preprocessing
# ============================================================
def load_and_clean(csv_path: str | Path) -> pd.DataFrame:
    """
    Load raw CSV and return a cleaned DataFrame ready for modelling.

    Steps:
        1. Drop ID / Name / SSN / Customer_ID / Month columns
        2. Clean numeric columns (remove stray underscores, symbols)
        3. Parse Credit_History_Age into months
        4. Encode categorical features
        5. Fill remaining NaN with column median (numeric) or mode (categorical)
    """
    df = pd.read_csv(csv_path)

    # Drop columns that carry no predictive value
    drop_cols = [c for c in ("ID", "Customer_ID", "Month", "Name", "SSN",
                             "Type_of_Loan") if c in df.columns]
    df = df.drop(columns=drop_cols)

    # --- Numeric cleaning ---
    for col in config.NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = _clean_numeric(df[col])

    # --- Credit history → months ---
    if "Credit_History_Age" in df.columns:
        df["Credit_History_Age"] = _parse_credit_history_months(
            df["Credit_History_Age"],
        )

    # --- Categorical encoding ---
    label_encoders: dict[str, LabelEncoder] = {}
    for col in config.CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip().replace("", np.nan)
        le = LabelEncoder()
        mask = df[col].notna()
        df.loc[mask, col] = le.fit_transform(df.loc[mask, col])
        df[col] = pd.to_numeric(df[col], errors="coerce")
        label_encoders[col] = le

    # --- Fill missing values ---
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    return df


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature columns used by the model."""
    return config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES + [
        "Credit_History_Age",
    ]


def prepare_splits(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    seed: int = config.SEED,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Split a cleaned DataFrame into train / val / test numpy arrays.

    Returns:
        {"train": (X_train, y_train),
         "val":   (X_val,   y_val),
         "test":  (X_test,  y_test)}
    """
    feature_cols = [c for c in get_feature_columns() if c in df.columns]
    target_col = "Credit_Score"

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].map(config.LABEL_MAP).values.astype(np.int64)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )

    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=relative_val, random_state=seed, stratify=y_train_full,
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "scaler": scaler,
        "feature_names": feature_cols,
    }
