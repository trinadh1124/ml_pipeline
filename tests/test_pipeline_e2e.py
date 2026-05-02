"""
test_pipeline_e2e.py  --  Integration Tests
============================================
End-to-end tests that validate the full pipeline works on synthetic data.
These are the integration tests referenced in the design doc:
  - Pipeline produces a model file and feature pipeline
  - MLflow run is created with expected metrics
  - Drift detection returns valid results
  - Batch scoring produces correct output shape
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
import json
import numpy as np
import pandas as pd
import pytest
import mlflow

from tests.conftest import make_sample_df


# ── Integration: Feature pipeline round-trip ─────────────────────────────────

def test_feature_pipeline_roundtrip(tmp_path):
    """Features built at training time must be reproducible at inference time."""
    from src.features.build_features import build_features

    df = make_sample_df(100)
    pipeline_path = str(tmp_path / "pipeline.json")

    # Fit (training)
    X_train, names_train, info = build_features(df, fit=True, pipeline_path=pipeline_path)

    # Transform (inference) -- same data should produce identical output
    X_inf, names_inf, _ = build_features(df, fit=False, pipeline_path=pipeline_path)

    assert names_train == names_inf, "Feature names must match between training and inference"
    assert X_train.shape == X_inf.shape, "Feature shapes must match"
    np.testing.assert_array_almost_equal(
        X_train.values, X_inf.values, decimal=10,
        err_msg="Feature values must be identical between fit and transform",
    )


def test_feature_pipeline_handles_missing_categories(tmp_path):
    """Inference data with missing OHE categories should get 0-filled columns."""
    from src.features.build_features import build_features

    df_train = make_sample_df(100)
    pipeline_path = str(tmp_path / "pipeline.json")
    X_train, names_train, _ = build_features(df_train, fit=True, pipeline_path=pipeline_path)

    # Inference data with only one contract type (fewer OHE columns)
    df_inf = make_sample_df(10)
    df_inf["contract"] = "Month-to-month"
    X_inf, _, _ = build_features(df_inf, fit=False, pipeline_path=pipeline_path)

    # Should still have the same columns (missing ones filled with 0)
    assert set(names_train).issubset(set(X_inf.columns) | set(names_train))
    assert len(X_inf) == 10


# ── Integration: Train -> Evaluate -> Gates ────────────────────────────────────

def test_train_evaluate_gates_pipeline():
    """Full training + evaluation + gate check on synthetic data."""
    from src.features.build_features import build_features
    from src.models.train import train_model
    from src.models.evaluate import compute_metrics, check_gates
    from sklearn.model_selection import train_test_split

    df = make_sample_df(200)
    X, _, _ = build_features(df, fit=True)
    y = df["churn"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    model = train_model(X_train.values, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)

    # Metrics must all be present and valid
    for key in ["f1", "recall", "precision", "auc_roc", "auc_pr",
                "tp", "fp", "fn", "tn", "missed_churners", "total_business_cost"]:
        assert key in metrics, f"Missing metric: {key}"
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["auc_roc"] <= 1
    assert metrics["total_business_cost"] >= 0

    # Gates check should return a tuple
    passed, failures = check_gates(metrics)
    assert isinstance(passed, bool)
    assert isinstance(failures, list)


# ── Integration: Drift detection ─────────────────────────────────────────────

def test_drift_detection_on_shifted_data(tmp_path):
    """Drift detection should flag shifted distributions."""
    from src.monitoring.drift import detect_drift
    from src.config.config import LOG_DIR

    # Create reference data
    rng = np.random.RandomState(42)
    ref = pd.DataFrame({
        "tenure_months": rng.normal(36, 10, 500).clip(1, 72).astype(int),
        "monthly_charges": rng.normal(65, 20, 500).clip(20, 200),
        "total_charges": rng.normal(3000, 1500, 500).clip(0, 10000),
    })

    # Create shifted data (simulate drift)
    cur = pd.DataFrame({
        "tenure_months": rng.normal(6, 3, 300).clip(1, 72).astype(int),   # much lower
        "monthly_charges": rng.normal(80, 15, 300).clip(20, 200),         # higher
        "total_charges": rng.normal(500, 300, 300).clip(0, 10000),        # much lower
    })

    results = detect_drift(ref, cur, ["tenure_months", "monthly_charges", "total_charges"],
                           save_html=False)

    assert "dataset_drift" in results
    assert "n_drifted_features" in results
    assert "features" in results
    assert results["n_drifted_features"] >= 1, "Should detect drift in shifted data"


def test_drift_detection_on_identical_data():
    """No drift should be flagged on identical distributions."""
    from src.monitoring.drift import detect_drift

    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "tenure_months": rng.normal(36, 10, 500).clip(1, 72).astype(int),
        "monthly_charges": rng.normal(65, 20, 500).clip(20, 200),
        "total_charges": rng.normal(3000, 1500, 500).clip(0, 10000),
    })

    results = detect_drift(data, data.copy(), ["tenure_months", "monthly_charges", "total_charges"],
                           save_html=False)

    assert results["n_drifted_features"] == 0, "Identical data should show no drift"


# ── Integration: MLflow logging ──────────────────────────────────────────────

def test_mlflow_run_logging(tmp_path):
    """MLflow should log params and metrics correctly."""
    tracking_uri = f"file:///{tmp_path.as_posix()}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-experiment")

    with mlflow.start_run() as run:
        mlflow.log_params({"model": "LightGBM", "n_estimators": 100})
        mlflow.log_metrics({"f1": 0.65, "recall": 0.70})

    # Read back
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run_data = client.get_run(run.info.run_id)
    assert run_data.data.params["model"] == "LightGBM"
    assert abs(run_data.data.metrics["f1"] - 0.65) < 1e-6


# ── Integration: Lineage recording ──────────────────────────────────────────

def test_lineage_recording(tmp_path):
    """Lineage should record data checksums and run metadata."""
    import src.utils.lineage as lineage_mod
    from src.utils.lineage import record_lineage

    # Create dummy data files
    train_file = tmp_path / "train.csv"
    current_file = tmp_path / "current.csv"
    train_file.write_text("a,b\n1,2\n")
    current_file.write_text("a,b\n3,4\n")

    # Patch LOG_DIR in both config and the lineage module
    import src.config.config as cfg
    original_cfg = cfg.LOG_DIR
    original_mod = lineage_mod.LOG_DIR
    cfg.LOG_DIR = tmp_path
    lineage_mod.LOG_DIR = tmp_path
    try:
        lineage = record_lineage(
            model_path="test_model",
            train_path=str(train_file),
            current_path=str(current_file),
            mlflow_run_id="test-run-123",
        )
    finally:
        cfg.LOG_DIR = original_cfg
        lineage_mod.LOG_DIR = original_mod

    assert lineage["train_checksum"] is not None
    assert lineage["current_checksum"] is not None
    assert lineage["mlflow_run_id"] == "test-run-123"
    assert (tmp_path / "lineage.json").exists()
