"""
Model factory — returns a scikit-learn compatible classifier.

Supported models:
    - logistic_regression
    - random_forest
    - xgboost
    - lightgbm
    - catboost
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import config

# Optional imports — installed via requirements.txt
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


ALL_MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
]


def build_model(name: str = "lightgbm", seed: int = config.SEED):
    """
    Build and return a classifier by name.

    Args:
        name: One of ALL_MODELS.
        seed: Random seed for reproducibility.

    Returns:
        A scikit-learn-compatible estimator.
    """
    if name == "logistic_regression":
        return LogisticRegression(
            **config.LOGISTIC_REGRESSION_PARAMS,
            random_state=seed,
            n_jobs=config.N_JOBS,
        )

    if name == "random_forest":
        return RandomForestClassifier(
            **config.RANDOM_FOREST_PARAMS,
            random_state=seed,
            n_jobs=config.N_JOBS,
        )

    if name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        return XGBClassifier(
            **config.XGBOOST_PARAMS,
            random_state=seed,
            n_jobs=config.N_JOBS,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )

    if name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
        return LGBMClassifier(
            **config.LIGHTGBM_PARAMS,
            random_state=seed,
            n_jobs=config.N_JOBS,
        )

    if name == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed. Run: pip install catboost")
        return CatBoostClassifier(
            **config.CATBOOST_PARAMS,
            random_seed=seed,
        )

    raise ValueError(
        f"Unknown model: {name}. Choose from {ALL_MODELS}"
    )
