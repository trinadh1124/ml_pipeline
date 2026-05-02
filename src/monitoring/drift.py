"""
drift.py  —  Drift Detection with Evidently
=============================================
MODULE 2 KEY TOOL: Evidently replaces the manual drift detection from Module 1.

In Module 1:
    shift = abs(train_mean - current_mean) / train_mean
    if shift > 0.10: flag_drift()

Now Evidently generates proper statistical drift reports with:
    - Per-column statistical tests (KS, chi-squared, etc.)
    - HTML reports for visual inspection
    - Programmatic access to drift results
    - Dataset-level and column-level drift detection
"""

import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from src.config.config import LOG_DIR, NUMERIC_FEATURES, BINARY_FEATURES


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str] | None = None,
    save_html: bool = True,
    report_name: str = "drift_report.html",
) -> dict:
    """
    Run Evidently data drift detection.

    Args:
        reference: Training data (baseline distribution).
        current: Post-deployment data.
        feature_cols: Columns to check. If None, uses numeric + binary.
        save_html: Save HTML drift report.

    Returns:
        dict with drift results per feature and overall drift flag.
    """
    if feature_cols is None:
        feature_cols = NUMERIC_FEATURES + BINARY_FEATURES

    # Keep only shared columns
    shared = [c for c in feature_cols if c in reference.columns and c in current.columns]
    ref = reference[shared].copy()
    cur = current[shared].copy()

    # Build column mapping
    num_cols = [c for c in NUMERIC_FEATURES if c in shared]
    cat_cols = [c for c in shared if c not in num_cols]
    column_mapping = ColumnMapping(
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )

    # Run drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur, column_mapping=column_mapping)

    # Save HTML report
    if save_html:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        html_path = LOG_DIR / report_name
        report.save_html(str(html_path))
        print(f"  [evidently] Drift report saved -> {html_path}")

    # Extract results
    result_dict = report.as_dict()
    metrics = result_dict.get("metrics", [])

    # Parse drift results
    drift_results = {
        "dataset_drift": False,
        "n_drifted_features": 0,
        "features": {},
    }

    for metric in metrics:
        metric_result = metric.get("result", {})
        if "drift_by_columns" in metric_result:
            drift_by_col = metric_result["drift_by_columns"]
            drift_results["dataset_drift"] = metric_result.get("dataset_drift", False)
            drift_results["n_drifted_features"] = metric_result.get(
                "number_of_drifted_columns", 0
            )
            for col_name, col_info in drift_by_col.items():
                drift_results["features"][col_name] = {
                    "drifted": col_info.get("drift_detected", False),
                    "statistic": col_info.get("stattest_name", "unknown"),
                    "p_value": col_info.get("drift_score", 1.0),
                }

    return drift_results


def print_drift_report(drift_results: dict):
    """Pretty-print the drift detection results."""
    print(f"\n{'='*60}")
    print(f"  DRIFT DETECTION REPORT (Evidently)")
    print(f"{'='*60}")
    print(f"  Dataset drift detected: {'YES' if drift_results['dataset_drift'] else 'NO'}")
    print(f"  Drifted features: {drift_results['n_drifted_features']}")
    print()
    for feat, info in drift_results["features"].items():
        status = "DRIFTED" if info["drifted"] else "stable"
        print(f"  {feat:25s}  {status:8s}  "
              f"(p={info['p_value']:.4f}, test={info['statistic']})")
    print(f"{'='*60}\n")
