"""
run_pipeline.py  -  Pipeline Orchestrator
=========================================
12 stages, same structure as Module 1, but wired to production tools.
"""

import json
import time

import mlflow
import numpy as np
from sklearn.model_selection import train_test_split

from src.config.config import (
    BATCH_PATHS,
    LOG_DIR,
    DRIFT_FEATURES,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    PRODUCTION_THRESHOLD,
    SEED,
    TARGET,
    TEST_SIZE,
    TRAIN_CSV,
)
from src.data.loader import load_current, load_current_stress, load_train
from src.data.prepare import describe_split
from src.data.validate import validate_dataframe
from src.features.build_features import build_features
from src.models.evaluate import (
    check_gates,
    compute_metrics,
    find_optimal_threshold,
    predict_at_threshold,
    print_evaluation,
    slice_analysis,
)
from src.models.explain import explain_model
from src.models.train import load_best_model_config, train_model
from src.monitoring.drift import detect_drift, print_drift_report
from src.registry.versioning import (
    load_production_model,
    promote_to_production,
    promote_to_staging,
    register_model,
)
from src.tracking.experiment import (
    get_run_id,
    log_artifact,
    log_metrics,
    log_model,
    log_params,
    setup_mlflow,
    start_run,
)
from src.utils.lineage import record_lineage


def _resolve_production_settings(y_train: np.ndarray) -> dict:
    """Load tuned settings if available, otherwise fall back to LightGBM defaults."""
    selected = load_best_model_config() or {}
    model_class = selected.get("model_class", "LGBMClassifier")
    params = selected.get("params", {})

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    default_spw = (n_neg / n_pos) * 1.5 if n_pos > 0 else 1.0

    threshold_cost = float(PRODUCTION_THRESHOLD)
    threshold_f1 = float(selected.get("threshold_f1", 0.50))

    return {
        "model_class": model_class,
        "params": params,
        "threshold_cost": threshold_cost,
        "threshold_f1": threshold_f1,
        "scale_pos_weight": params.get("scale_pos_weight", default_spw),
        "selected_config_found": bool(selected),
    }


def run_pipeline(monitor_batch: str = "current"):
    """Execute the full 12-stage MLOps pipeline."""
    t0 = time.time()
    if monitor_batch not in BATCH_PATHS:
        valid = ", ".join(BATCH_PATHS)
        raise ValueError(f"Unknown monitor_batch '{monitor_batch}'. Use one of: {valid}")
    deploy_run = monitor_batch == "current"

    print("\n" + "=" * 70)
    print("  STAGE 0 - Initialisation")
    print("=" * 70)
    np.random.seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    setup_mlflow()
    print(f"  Random seed: {SEED}")

    print("\n" + "=" * 70)
    print("  STAGE 1 - Data Loading")
    print("=" * 70)
    train_df = load_train()
    current_df = load_current(batch_name=monitor_batch)
    stress_df = load_current_stress()
    print(f"  Train:   {len(train_df):,} rows (historical batch available at deployment)")
    print(f"  Current: {len(current_df):,} rows (selected post-deployment '{monitor_batch}' batch)")
    print(f"  Stress:  {len(stress_df):,} rows (available shifted monitoring batch)")

    print("\n" + "=" * 70)
    print("  STAGE 2 - Data Validation (Pandera)")
    print("=" * 70)
    train_df = validate_dataframe(train_df, "train")
    current_df = validate_dataframe(current_df, monitor_batch)
    stress_df = validate_dataframe(stress_df, "current_stress")
    describe_split(train_df, load_current(batch_name="current"), stress_df)

    print("\n" + "=" * 70)
    print("  STAGE 3 - Leakage-Safe Split + Feature Engineering")
    print("=" * 70)
    train_fit_df, train_eval_df = train_test_split(
        train_df, test_size=TEST_SIZE, random_state=SEED, stratify=train_df[TARGET],
    )
    train_fit_df = train_fit_df.reset_index(drop=True)
    train_eval_df = train_eval_df.reset_index(drop=True)

    X_train, feature_names, _ = build_features(train_fit_df, fit=True)
    X_test, _, _ = build_features(
        train_eval_df, fit=False, pipeline_path=str(MODEL_DIR / "feature_pipeline.json"),
    )
    X_current_eval, _, _ = build_features(
        current_df, fit=False, pipeline_path=str(MODEL_DIR / "feature_pipeline.json"),
    )
    y_train = train_fit_df[TARGET].values
    y_test = train_eval_df[TARGET].values
    print(f"  Features: {len(feature_names)} columns after OHE")
    print("  Feature pipeline fit only on the training fold to avoid hold-out leakage")

    settings = _resolve_production_settings(y_train)
    print(f"  Production estimator: {settings['model_class']}")
    print(f"  Production threshold: {settings['threshold_cost']:.2f} (fixed learner-facing threshold)")

    print("\n" + "=" * 70)
    print("  STAGE 4 - Train/Test Split Summary")
    print("=" * 70)
    print(f"  Train: {len(X_train)} samples ({y_train.mean():.1%} churn)")
    print(f"  Test:  {len(X_test)} samples ({y_test.mean():.1%} churn)")

    print("\n" + "=" * 70)
    print("  STAGE 5 - Model Training (MLflow tracked)")
    print("=" * 70)
    with start_run(run_name="pipeline-run") as run:
        run_id = get_run_id()
        print(f"  [mlflow] Run ID: {run_id}")

        log_params({
            "model_class": settings["model_class"],
            "test_size": TEST_SIZE,
            "seed": SEED,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "n_features": len(feature_names),
            "threshold_cost": settings["threshold_cost"],
            "threshold_f1": settings["threshold_f1"],
            "selected_config_found": settings["selected_config_found"],
            **settings["params"],
        })

        eval_model = train_model(
            X_train.values,
            y_train,
            scale_pos_weight=settings["scale_pos_weight"],
            params=settings["params"],
            model_class=settings["model_class"],
        )

        print("\n" + "=" * 70) 
        print("  STAGE 6 - Pre-Deployment Evaluation")
        print("=" * 70)
        y_prob = eval_model.predict_proba(X_test)[:, 1]
        y_pred_default = predict_at_threshold(y_prob, threshold=0.50)
        metrics = compute_metrics(y_test, y_pred_default, y_prob, threshold=0.50)
        print_evaluation(metrics, "Hold-out Test Set (default threshold)")

        cost_t, f1_t, threshold_report = find_optimal_threshold(
            y_test, y_prob, min_precision=0.40,
        )
        fixed_t = float(PRODUCTION_THRESHOLD)
        prod_preds = predict_at_threshold(y_prob, threshold=fixed_t)
        prod_metrics = compute_metrics(y_test, prod_preds, y_prob, threshold=fixed_t)
        print_evaluation(prod_metrics, "Hold-out Test Set (production threshold)")

        log_metrics({
            "f1": metrics["f1"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "accuracy": metrics["accuracy"],
            "auc_roc": metrics["auc_roc"],
            "auc_pr": metrics["auc_pr"],
            "missed_churners": metrics["missed_churners"],
            "total_business_cost": metrics["total_business_cost"],
            "prod_threshold_cost": fixed_t,
            "prod_f1": prod_metrics["f1"],
            "prod_recall": prod_metrics["recall"],
            "prod_precision": prod_metrics["precision"],
            "prod_total_business_cost": prod_metrics["total_business_cost"],
        })

        print("  Slice analysis by contract type:")
        if "contract_One year" in X_test.columns:
            slices = slice_analysis(eval_model, X_test, y_test, "contract_One year")
            print(slices.to_string(index=False))

        print("\n" + "=" * 70)
        print("  STAGE 6b - Model Interpretability (SHAP)")
        print("=" * 70)
        try:
            shap_results = explain_model(
                eval_model, X_test, list(X_test.columns), save_dir=str(LOG_DIR),
            )
            log_artifact(str(shap_results["summary_plot"]))
            log_artifact(str(shap_results["bar_plot"]))
        except Exception as e:
            print(f"  [shap] Skipped - {e}")

        print("\n" + "=" * 70)
        print("  STAGE 7 - Deployment Gates")
        print("=" * 70)
        passed, failures = check_gates(metrics)
        if not passed:
            for failure in failures:
                print(f"    FAIL: {failure}")
            mlflow.set_tag("gate_status", "failed")
            print("\n  Pipeline stopped. Model not deployed.")
            return metrics

        print("  ALL GATES PASSED - refitting production model on full training data")
        mlflow.set_tag("gate_status", "passed")

        # Refit on the full historical training set after validation passes.
        X_full, full_feature_names, _ = build_features(train_df, fit=True)
        full_y = train_df[TARGET].values
        production_model = train_model(
            X_full.values,
            full_y,
            scale_pos_weight=settings["scale_pos_weight"],
            params=settings["params"],
            model_class=settings["model_class"],
        )
        X_current, _, _ = build_features(
            current_df, fit=False, pipeline_path=str(MODEL_DIR / "feature_pipeline.json"),
        )

        if deploy_run:
            # Persist updated production settings only for the stable deployment path.
            best_config = load_best_model_config() or {}
            best_config.update({
                "model_class": settings["model_class"],
                "params": settings["params"],
                "threshold_cost": round(float(fixed_t), 2),
                "threshold_f1": round(float(f1_t), 2),
                "holdout_metrics_default": {
                    "f1": round(float(metrics["f1"]), 4),
                    "recall": round(float(metrics["recall"]), 4),
                    "precision": round(float(metrics["precision"]), 4),
                    "auc_roc": round(float(metrics["auc_roc"]), 4),
                    "auc_pr": round(float(metrics["auc_pr"]), 4),
                },
                "holdout_metrics_production": {
                    "f1": round(float(prod_metrics["f1"]), 4),
                    "recall": round(float(prod_metrics["recall"]), 4),
                    "precision": round(float(prod_metrics["precision"]), 4),
                    "total_business_cost": int(prod_metrics["total_business_cost"]),
                },
            })
            with open(MODEL_DIR / "best_params.json", "w") as f:
                json.dump(best_config, f, indent=2)

            print("\n" + "=" * 70)
            print("  STAGE 8 - MLflow Model Logging")
            print("=" * 70)
            log_model(production_model, X_full, model_name="model")
            log_artifact(str(MODEL_DIR / "feature_pipeline.json"))
            log_artifact(str(MODEL_DIR / "best_params.json"))

            print("\n" + "=" * 70)
            print("  STAGE 9 - Model Registry (MLflow)")
            print("=" * 70)
            try:
                version = register_model(run_id)
                promote_to_staging(version)
                promote_to_production(version)
            except Exception as e:
                print(f"  [registry] Registration failed: {e}")
                print("  Model is logged in MLflow but not registered. Check MLflow URI.")
        else:
            print("\n" + "=" * 70)
            print("  STAGE 8/9 - Stress Monitoring Mode")
            print("=" * 70)
            print("  Stress runs do not update best_params.json or replace production.")
            try:
                production_model = load_production_model()
                print("  Loaded the existing production model for stress monitoring.")
            except Exception as e:
                print(f"  [registry] Could not load existing production model: {e}")
                print("  Falling back to the historical refit for this monitoring-only run.")

        print("\n" + "=" * 70)
        print("  STAGE 10 - Drift Detection (Evidently)")
        print("=" * 70)
    try:
        drift_cols = [c for c in DRIFT_FEATURES if c in train_df.columns and c in current_df.columns]
        drift_results = detect_drift(
            train_df[drift_cols],
            current_df[drift_cols],
            feature_cols=drift_cols,
            report_name=f"drift_report_{monitor_batch}.html",
        )
        print_drift_report(drift_results)
    except Exception as e:
        print(f"  [evidently] Drift detection failed: {e}")
        print("  Pipeline continues - drift report unavailable this run.")

    print("\n" + "=" * 70)
    print("  STAGE 11 - Batch Scoring (incoming production batch)")
    print("=" * 70)
    probabilities = production_model.predict_proba(X_current)[:, 1]
    predictions = predict_at_threshold(probabilities, threshold=fixed_t)
    scored = current_df.copy()
    scored["churn_prediction"] = predictions
    scored["churn_probability"] = probabilities
    scored.to_csv(LOG_DIR / f"scored_{monitor_batch}.csv", index=False)
    print(f"  Scored {len(scored)} customers")
    print(f"  Predicted churners: {predictions.sum()} ({predictions.mean():.1%})")
    print(f"  Actual churners:    {current_df[TARGET].sum()} ({current_df[TARGET].mean():.1%})")

    y_current = current_df[TARGET].values
    current_metrics = compute_metrics(y_current, predictions, probabilities, threshold=fixed_t)
    print_evaluation(current_metrics, "Post-Deployment Cohort (production threshold)")

    metrics_path = LOG_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "monitor_batch": monitor_batch,
            "holdout_default": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            "holdout_production": {k: v for k, v in prod_metrics.items() if isinstance(v, (int, float))},
            f"{monitor_batch}_production": {k: v for k, v in current_metrics.items() if isinstance(v, (int, float))},
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("  STAGE 12 - Lineage Recording")
    print("=" * 70)
    record_lineage(
        model_path="mlflow_registry",
        train_path=str(TRAIN_CSV),
        current_path=str(BATCH_PATHS[monitor_batch]),
        mlflow_run_id=run_id,
        extra={
            "metrics": metrics,
            "production_metrics": prod_metrics,
            "production_threshold": fixed_t,
            "pipeline_version": "module2-v2.0",
            "monitor_batch": monitor_batch,
        },
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE - {elapsed:.1f}s")
    print(f"  MLflow UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print(f"{'=' * 70}\n")

    return prod_metrics
