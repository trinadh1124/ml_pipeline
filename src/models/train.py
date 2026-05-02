"""
train.py  -  Model Training
===========================
Builds the selected production estimator from the saved model config.
"""

import json

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config.config import LGBM_PARAMS, MODEL_DIR, SEED

BEST_PARAMS_PATH = MODEL_DIR / "best_params.json"


def load_best_model_config() -> dict | None:
    """Load the selected production model config if it exists."""
    if not BEST_PARAMS_PATH.exists():
        return None
    with open(BEST_PARAMS_PATH) as f:
        return json.load(f)


def build_estimator(
    model_class: str = "LGBMClassifier",
    params: dict | None = None,
    scale_pos_weight: float | None = None,
):
    """Instantiate the estimator with consistent training defaults."""
    params = dict(params or {})

    if model_class == "LGBMClassifier":
        p = {**LGBM_PARAMS, **params}
        if scale_pos_weight is not None and "scale_pos_weight" not in p:
            p["scale_pos_weight"] = scale_pos_weight
        p.setdefault("random_state", SEED)
        return LGBMClassifier(**p)

    if model_class == "RandomForestClassifier":
        p = {"class_weight": "balanced", "random_state": SEED, **params}
        return RandomForestClassifier(**p)

    if model_class == "XGBClassifier":
        p = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "max_delta_step": 0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": SEED,
            **params,
        }
        if scale_pos_weight is not None and "scale_pos_weight" not in p:
            p["scale_pos_weight"] = scale_pos_weight
        return XGBClassifier(**p)

    if model_class == "GradientBoostingClassifier":
        p = {"random_state": SEED, **params}
        return GradientBoostingClassifier(**p)

    if model_class == "LogisticRegression":
        p = {
            "class_weight": "balanced",
            "max_iter": 5000,
            "random_state": SEED,
            **params,
        }
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**p)),
        ])

    raise ValueError(f"Unsupported model_class: {model_class}")


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float | None = None,
    params: dict | None = None,
    model_class: str = "LGBMClassifier",
):
    """
    Train the selected production classifier.

    Args:
        X_train: Feature matrix.
        y_train: Binary target.
        scale_pos_weight: If None, computed from class ratio.
        params: Override the saved/default estimator params.
        model_class: Estimator class name to instantiate.

    Returns:
        Fitted estimator.
    """
    if scale_pos_weight is None:
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = (n_neg / n_pos) * 1.5 if n_pos > 0 else 1.0

    model = build_estimator(
        model_class=model_class,
        params=params,
        scale_pos_weight=scale_pos_weight,
    )
    fit_kwargs = {}
    if model_class == "LGBMClassifier":
        fit_kwargs = {"eval_set": [(X_train, y_train)], "callbacks": []}

    model.fit(X_train, y_train, **fit_kwargs)

    if model_class == "LGBMClassifier":
        n_estimators = model.get_params()["n_estimators"]
        print(f"  [train] LightGBM fitted: {n_estimators} trees, "
              f"scale_pos_weight={scale_pos_weight:.2f}")
    else:
        print(f"  [train] {model_class} fitted on {len(X_train):,} rows")
    return model
