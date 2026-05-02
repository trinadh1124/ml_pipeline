"""
tuning.py  --  Dynamic Hyperparameter Tuning with Optuna
=========================================================
Tunes whichever model is passed in -- not hardcoded to LightGBM.
Called by model_selection.py on each of the top-3 models.

Key design decisions:
  - Uses 3-fold stratified CV for evaluation (not a single hold-out split)
    so estimates are stable with ~4k rows
  - LogisticRegression uses StandardScaler inside the objective
    (without scaling, lbfgs cannot converge -- wrong search results)
  - scale_pos_weight search range is dataset-specific
"""

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from src.config.config import SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)

CV_FOLDS = 3


def _make_objective(model_name, X_train, y_train, base_spw):

    def objective(trial):
        if model_name == "LightGBM":
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 800, step=50),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth":         trial.suggest_int("max_depth", 3, 10),
                "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "scale_pos_weight":  trial.suggest_float(
                    "scale_pos_weight", base_spw * 0.5, base_spw * 2.0,
                ),
                "random_state": SEED, "verbosity": -1,
            }
            model = LGBMClassifier(**params)

        elif model_name == "Random Forest":
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth":         trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "class_weight": "balanced", "random_state": SEED,
            }
            model = RandomForestClassifier(**params)

        elif model_name == "XGBoost (GBM)":
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 1000, step=50),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth":         trial.suggest_int("max_depth", 2, 8),
                "min_child_weight":  trial.suggest_float("min_child_weight", 1.0, 12.0),
                "gamma":             trial.suggest_float("gamma", 1e-8, 5.0, log=True),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "max_delta_step":    trial.suggest_int("max_delta_step", 0, 5),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "scale_pos_weight":  trial.suggest_float(
                    "scale_pos_weight", base_spw * 0.7, base_spw * 2.5,
                ),
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "n_jobs": -1,
                "random_state": SEED,
            }
            model = XGBClassifier(**params)

        elif model_name == "Logistic Regression":
            # StandardScaler inside the pipeline -- without scaling lbfgs diverges
            C = trial.suggest_float("C", 1e-3, 10.0, log=True)
            solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    C=C, solver=solver, max_iter=5000,
                    class_weight="balanced", random_state=SEED,
                )),
            ])
        else:
            raise ValueError(f"Unknown model: {model_name}")

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        return scores.mean()

    return objective


def tune_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    n_trials: int = 50,
    timeout: int = 300,
) -> tuple[dict, float]:
    """
    Bayesian hyperparameter search using Optuna with 3-fold CV.

    Args:
        model_name: One of 'LightGBM', 'Random Forest', 'XGBoost (GBM)',
                    'Logistic Regression'
        X_train, y_train: Full training data (CV splits internally).
        X_val, y_val:     Unused -- kept for API compatibility.
        n_trials:         Optuna trials (default 50).
        timeout:          Max seconds.

    Returns:
        (best_params dict, best_cv_f1)
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    base_spw = n_neg / n_pos if n_pos > 0 else 1.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(
        _make_objective(model_name, X_train, y_train, base_spw),
        n_trials=n_trials,
        timeout=timeout,
    )

    best = study.best_trial
    print(f"\n  {'-'*48}")
    print(f"  Optuna: {model_name} ({len(study.trials)} trials, {CV_FOLDS}-fold CV)")
    print(f"  {'-'*48}")
    print(f"  Best CV F1: {best.value:.4f}")
    for k, v in best.params.items():
        print(f"    {k:25s} = {v:.4f}" if isinstance(v, float) else f"    {k:25s} = {v}")
    print(f"  {'-'*48}")

    return best.params, best.value
