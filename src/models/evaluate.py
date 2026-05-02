"""
evaluate.py  -  Model Evaluation
================================
Computes metrics, checks deployment gates, and performs slice analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config.config import (
    COST_FN,
    COST_FP,
    GATE_AUC_PR,
    GATE_AUC_ROC,
    GATE_F1,
    GATE_RECALL,
)


def predict_at_threshold(y_prob: np.ndarray, threshold: float = 0.50) -> np.ndarray:
    """Convert probabilities to binary labels at a chosen threshold."""
    return (y_prob >= threshold).astype(int)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray | None,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> dict:
    """Compute all evaluation metrics."""
    if y_pred is None:
        y_pred = predict_at_threshold(y_prob, threshold=threshold)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "missed_churners": int(fn),
        "business_cost_fn": int(fn) * COST_FN,
        "business_cost_fp": int(fp) * COST_FP,
        "total_business_cost": int(fn) * COST_FN + int(fp) * COST_FP,
    }


def check_gates(metrics: dict) -> tuple[bool, list[str]]:
    """Check deployment gates. Returns (passed, list of failures)."""
    failures = []
    if metrics["f1"] < GATE_F1:
        failures.append(f"F1 {metrics['f1']:.3f} < {GATE_F1}")
    if metrics["recall"] < GATE_RECALL:
        failures.append(f"Recall {metrics['recall']:.3f} < {GATE_RECALL}")
    if metrics["auc_pr"] < GATE_AUC_PR:
        failures.append(f"AUC-PR {metrics['auc_pr']:.3f} < {GATE_AUC_PR}")
    if metrics["auc_roc"] < GATE_AUC_ROC:
        failures.append(f"AUC-ROC {metrics['auc_roc']:.3f} < {GATE_AUC_ROC}")

    return len(failures) == 0, failures


def print_evaluation(metrics: dict, title: str = "Evaluation"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'-'*50}")
    print(f"  {title}")
    print(f"{'-'*50}")
    print(f"  Threshold : {metrics.get('threshold', 0.50):.2f}")
    print(f"  Accuracy  : {metrics['accuracy']:.3f}")
    print(f"  F1        : {metrics['f1']:.3f}")
    print(f"  Recall    : {metrics['recall']:.3f}")
    print(f"  Precision : {metrics['precision']:.3f}")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.3f}")
    print(f"  AUC-PR    : {metrics['auc_pr']:.3f}")
    print(f"  Confusion : TP={metrics['tp']} FP={metrics['fp']} "
          f"FN={metrics['fn']} TN={metrics['tn']}")
    print(f"  Missed churners : {metrics['missed_churners']} "
          f"(${metrics['business_cost_fn']:,} lost)")
    print(f"  Wasted offers   : {metrics['fp']} "
          f"(${metrics['business_cost_fp']:,} wasted)")
    print(f"  Total cost      : ${metrics['total_business_cost']:,}")
    print(f"{'-'*50}\n")


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: int = COST_FN,
    cost_fp: int = COST_FP,
    min_precision: float | None = None,
) -> tuple[float, float, dict]:
    """
    Find cost-optimal and F1-optimal thresholds.
    """
    thresholds = np.arange(0.05, 0.70, 0.01)
    best_cost, best_cost_t = float("inf"), 0.5
    best_f1, best_f1_t = 0.0, 0.5
    sweep = []

    constrained_best_cost, constrained_best_t = float("inf"), None

    for t in thresholds:
        preds = predict_at_threshold(y_prob, threshold=float(t))
        if preds.sum() == 0:
            continue
        fn = int(((y_true == 1) & (preds == 0)).sum())
        fp = int(((y_true == 0) & (preds == 1)).sum())
        cost = fn * cost_fn + fp * cost_fp
        f1 = f1_score(y_true, preds, zero_division=0)
        precision = precision_score(y_true, preds, zero_division=0)
        sweep.append({
            "threshold": round(float(t), 2),
            "cost": cost,
            "f1": f1,
            "fn": fn,
            "fp": fp,
            "precision": precision,
        })
        if cost < best_cost:
            best_cost, best_cost_t = cost, float(t)
        if min_precision is not None and precision >= min_precision and cost < constrained_best_cost:
            constrained_best_cost, constrained_best_t = cost, float(t)
        if f1 > best_f1:
            best_f1, best_f1_t = f1, float(t)

    if constrained_best_t is not None:
        best_cost_t = constrained_best_t
        best_cost = constrained_best_cost

    default_cost = next(
        (s["cost"] for s in sweep if abs(s["threshold"] - 0.50) < 1e-9),
        None,
    )
    print(f"  Threshold optimisation (FN=${cost_fn:,}, FP=${cost_fp:,}):")
    print(f"    Cost-optimal  : t={best_cost_t:.2f}  -> total cost ${best_cost:,}")
    print(f"    F1-optimal    : t={best_f1_t:.2f}  -> F1={best_f1:.4f}")
    if default_cost is not None:
        print(f"    Default (0.50): cost ${default_cost:,}")

    return best_cost_t, best_f1_t, {
        "sweep": sweep,
        "cost_optimal": best_cost_t,
        "f1_optimal": best_f1_t,
        "best_cost": best_cost,
        "min_precision": min_precision,
    }


def slice_analysis(model, X_test: pd.DataFrame, y_test: np.ndarray,
                   slice_col: str) -> pd.DataFrame:
    """Evaluate model performance on slices of a categorical feature."""
    results = []
    for value in sorted(X_test[slice_col].unique()):
        mask = X_test[slice_col] == value
        if mask.sum() < 10:
            continue
        y_sub = y_test[mask]
        pred = model.predict(X_test[mask])
        results.append({
            "slice": f"{slice_col}={value}",
            "n": int(mask.sum()),
            "churn_rate": float(y_sub.mean()),
            "f1": f1_score(y_sub, pred, zero_division=0),
            "recall": recall_score(y_sub, pred, zero_division=0),
        })
    return pd.DataFrame(results)
