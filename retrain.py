"""
retrain.py  --  Drift-Triggered Retraining
===========================================
Evaluates normal or stress post-deployment batches and recommends the
appropriate retraining response.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

from src.config.config import (
    DRIFT_FEATURES,
    LGBM_PARAMS,
    MODEL_DIR,
    MLFLOW_TRACKING_URI,
    PRODUCTION_THRESHOLD,
    SEED,
    TARGET,
)
from src.data.loader import load_current, load_train
from src.data.validate import validate_dataframe
from src.features.build_features import build_features
from src.models.evaluate import (
    check_gates,
    compute_metrics,
    find_optimal_threshold,
    predict_at_threshold,
    print_evaluation,
)
from src.models.train import train_model
from src.monitoring.drift import detect_drift, print_drift_report
from src.registry.versioning import register_model, promote_to_staging, promote_to_production
from src.tracking.experiment import setup_mlflow, log_params, log_metrics, log_model, get_run_id


HIGH_IMPORTANCE_DRIFT = ["tenure_months", "monthly_charges", "total_charges"]


def evaluate_retrain_need(
    drift_results: dict,
    baseline_f1: float | None = None,
    current_f1: float | None = None,
) -> tuple[bool, str | None, list[str], list[str]]:
    """Evaluate trigger rules. Returns (should_retrain, recommended_strategy, reasons)."""
    reasons: list[str] = []
    observations: list[str] = []

    for feat in HIGH_IMPORTANCE_DRIFT:
        if drift_results["features"].get(feat, {}).get("drifted"):
            reasons.append(f"Rule 1: {feat} drifted (high importance)")

    if drift_results["n_drifted_features"] >= 2:
        reasons.append(f"Rule 2: {drift_results['n_drifted_features']} features drifted")

    if baseline_f1 is not None and current_f1 is not None:
        drop = baseline_f1 - current_f1
        if drop > 0.05:
            if drift_results["dataset_drift"] or drift_results["n_drifted_features"] > 0:
                reasons.append(f"Rule 3: F1 dropped {drop:.3f} (>0.05)")
            else:
                observations.append(
                    f"F1 dropped {drop:.3f} (>0.05) but no input drift was detected; monitor only."
                )

    should_retrain = bool(reasons)
    recommended = None
    if should_retrain:
        recommended = "reexperiment" if any("Rule 3" in reason for reason in reasons) else "mixed"
    return should_retrain, recommended, reasons, observations


def retrain_finetune(model, X_new, y_new):
    """Warm-start: add trees on top of existing model using only new data."""
    print("\n  Strategy: FINETUNE (warm-start on selected batch only)")
    new_model = LGBMClassifier(
        **{**LGBM_PARAMS, "n_estimators": 150, "init_model": model},
    )
    n_neg = int((y_new == 0).sum())
    n_pos = int((y_new == 1).sum())
    new_model.set_params(scale_pos_weight=n_neg / n_pos if n_pos > 0 else 1.0)
    new_model.fit(X_new, y_new)
    return new_model


def retrain_mixed(X_hist, y_hist, X_new, y_new, sample_frac: float = 0.5):
    """Pool historical sample + all new data. Train from scratch."""
    print(f"\n  Strategy: MIXED ({sample_frac:.0%} historical + 100% selected batch)")
    n_sample = int(len(X_hist) * sample_frac)
    idx = np.random.RandomState(SEED).choice(len(X_hist), n_sample, replace=False)
    X_combined = pd.concat([
        pd.DataFrame(X_hist).iloc[idx],
        pd.DataFrame(X_new),
    ], ignore_index=True)
    y_combined = np.concatenate([y_hist[idx], y_new])
    return train_model(X_combined.values, y_combined)


def retrain_reexperiment(X_combined, y_combined):
    """Full hyperparameter search on combined data."""
    print("\n  Strategy: REEXPERIMENT (hyperparameter search)")
    configs = [
        {"n_estimators": 300, "max_depth": 4, "num_leaves": 15},
        {"n_estimators": 500, "max_depth": 6, "num_leaves": 31},
        {"n_estimators": 700, "max_depth": 8, "num_leaves": 63},
        {"n_estimators": 500, "max_depth": 6, "num_leaves": 31, "learning_rate": 0.02},
    ]
    best_f1, best_model = 0, None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=SEED, stratify=y_combined,
    )
    for i, cfg in enumerate(configs):
        params = {**LGBM_PARAMS, **cfg}
        n_neg = int((y_tr == 0).sum())
        n_pos = int((y_tr == 1).sum())
        params["scale_pos_weight"] = n_neg / n_pos if n_pos > 0 else 1.0
        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        f1 = __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(y_val, model.predict(X_val))
        print(f"  Config {i+1}: F1={f1:.3f} (n_est={cfg['n_estimators']}, depth={cfg['max_depth']})")
        if f1 > best_f1:
            best_f1, best_model = f1, model
    print(f"  Best: F1={best_f1:.3f}")
    return best_model


def _load_baseline_f1() -> float | None:
    best_params_path = MODEL_DIR / "best_params.json"
    if not best_params_path.exists():
        return None
    with open(best_params_path) as f:
        payload = json.load(f)
    holdout = payload.get("holdout_metrics_production") or {}
    return holdout.get("f1")


def main():
    parser = argparse.ArgumentParser(description="Drift-triggered retraining")
    parser.add_argument("--batch", choices=["current", "stress"], default="current")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate only, don't retrain")
    parser.add_argument("--strategy", choices=["finetune", "mixed", "reexperiment"],
                        default=None, help="Retraining strategy")
    parser.add_argument("--force", action="store_true", help="Override no-trigger decision")
    args = parser.parse_args()

    print("="*70)
    print("  RETRAIN -- Drift-Triggered Retraining")
    print("="*70)

    setup_mlflow()

    train_df = validate_dataframe(load_train(), "historical")
    batch_df = validate_dataframe(load_current(batch_name=args.batch), f"{args.batch} batch")

    X_hist, _, _ = build_features(train_df, fit=True)
    y_hist = train_df[TARGET].values

    X_new, _, _ = build_features(
        batch_df, fit=False, pipeline_path=str(MODEL_DIR / "feature_pipeline.json"),
    )
    y_new = batch_df[TARGET].values

    drift_cols = [col for col in DRIFT_FEATURES if col in train_df.columns and col in batch_df.columns]
    drift_results = detect_drift(
        train_df[drift_cols],
        batch_df[drift_cols],
        drift_cols,
        report_name=f"drift_report_{args.batch}.html",
    )
    print_drift_report(drift_results)

    baseline_f1 = _load_baseline_f1()
    threshold = float(PRODUCTION_THRESHOLD)
    current_prob = None
    current_f1 = None
    try:
        from src.registry.versioning import load_production_model
        production_model = load_production_model()
        current_prob = production_model.predict_proba(X_new)[:, 1]
        current_pred = (current_prob >= threshold).astype(int)
        current_metrics = compute_metrics(y_new, current_pred, current_prob, threshold=threshold)
        current_f1 = current_metrics["f1"]
        print_evaluation(current_metrics, f"Production Model on '{args.batch}' batch")
    except Exception as exc:
        print(f"  [warn] Could not score current production model for F1 degradation: {exc}")

    should_retrain, recommended_strategy, reasons, observations = evaluate_retrain_need(
        drift_results, baseline_f1=baseline_f1, current_f1=current_f1,
    )
    print(f"  Batch under review: {args.batch}")
    print(f"  Should retrain: {'YES' if should_retrain else 'NO'}")
    if recommended_strategy:
        print(f"  Recommended strategy: {recommended_strategy}")
    for reason in reasons:
        print(f"    -> {reason}")
    for observation in observations:
        print(f"    -> {observation}")

    if args.dry_run:
        if should_retrain:
            print(f"\n  [dry-run] '{args.batch}' would trigger retraining via {recommended_strategy}.")
        else:
            print(f"\n  [dry-run] '{args.batch}' stays on the current production model.")
        return

    if not should_retrain and not args.force:
        print(f"\n  No retrain needed for '{args.batch}'. Use --force to override.")
        return

    X_new_train, X_new_val, y_new_train, y_new_val = train_test_split(
        X_new, y_new, test_size=0.3, random_state=SEED, stratify=y_new,
    )
    print(f"\n  Selected batch split: {len(X_new_train)} train, {len(X_new_val)} held-out")

    strategy = args.strategy or recommended_strategy or "mixed"
    with mlflow.start_run(run_name=f"retrain-{args.batch}-{strategy}"):
        log_params({
            "batch": args.batch,
            "strategy": strategy,
            "historical_rows": len(X_hist),
            "new_rows": len(X_new_train),
            "new_holdout": len(X_new_val),
        })

        if strategy == "finetune":
            from src.registry.versioning import load_production_model
            existing = load_production_model()
            model = retrain_finetune(existing, X_new_train.values, y_new_train)
        elif strategy == "mixed":
            model = retrain_mixed(X_hist.values, y_hist, X_new_train.values, y_new_train)
        else:
            X_comb = pd.concat([X_hist, X_new_train], ignore_index=True)
            y_comb = np.concatenate([y_hist, y_new_train])
            model = retrain_reexperiment(X_comb.values, y_comb)

        y_prob = model.predict_proba(X_new_val)[:, 1]
        cost_t, f1_t, _ = find_optimal_threshold(y_new_val, y_prob, min_precision=0.40)
        chosen_threshold = float(PRODUCTION_THRESHOLD)
        if chosen_threshold != cost_t:
            print(
                f"  Fixed production threshold applied: using {chosen_threshold:.2f} "
                f"instead of cost-optimal {cost_t:.2f}"
            )
        y_pred = predict_at_threshold(y_prob, threshold=chosen_threshold)
        metrics = compute_metrics(y_new_val, y_pred, y_prob, threshold=chosen_threshold)
        print_evaluation(
            metrics,
            f"Retrained Model ({strategy}) on '{args.batch}' hold-out (fixed production threshold)",
        )

        log_metrics({
            "f1": metrics["f1"],
            "recall": metrics["recall"],
            "auc_roc": metrics["auc_roc"],
            "auc_pr": metrics["auc_pr"],
            "precision": metrics["precision"],
            "missed_churners": metrics["missed_churners"],
            "threshold_cost": chosen_threshold,
            "threshold_f1": f1_t,
            "total_business_cost": metrics["total_business_cost"],
        })

        passed, failures = check_gates(metrics)
        if passed:
            log_model(model, X_new_val, model_name="model")
            run_id = get_run_id()
            version = register_model(run_id)
            promote_to_staging(version)
            promote_to_production(version)
            print(f"\n  Retrained model promoted to production (v{version})")
        else:
            print(f"\n  Gates failed -- retrained model NOT promoted")
            for failure in failures:
                print(f"    FAIL: {failure}")


if __name__ == "__main__":
    main()
