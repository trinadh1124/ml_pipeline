"""
batch_infer.py  —  Batch Inference
====================================
Scores the post-deployment customer cohort using the production model.

    python batch_infer.py
    python batch_infer.py --input data/current.csv --output data/scored.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import time
import pandas as pd
import mlflow

from src.config.config import (
    LOG_DIR, MODEL_DIR, TARGET,
    MLFLOW_TRACKING_URI, MLFLOW_MODEL_NAME,
)
from src.data.loader import resolve_batch_path
from src.features.build_features import build_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", choices=["current", "stress"], default="current")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = args.input or str(resolve_batch_path(args.batch))
    output_path = args.output or str(LOG_DIR / f"scored_{args.batch}.csv")

    print("="*70)
    print("  BATCH INFERENCE")
    print("="*70)

    # Load data
    df = pd.read_csv(input_path)
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    print(f"  Batch: {args.batch}")
    print(f"  Input: {input_path} ({len(df)} rows)")

    # Load production model
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        from src.registry.versioning import load_production_model
        model = load_production_model()
    except Exception as e:
        print(f"  MLflow load failed: {e}")
        print(f"  Falling back to local model...")
        import joblib
        model_files = sorted(MODEL_DIR.glob("*.pkl"))
        if not model_files:
            print("  ERROR: No model found. Run main.py first.")
            return
        model = joblib.load(model_files[-1])

    # Build features
    pipeline_path = str(MODEL_DIR / "feature_pipeline.json")
    X, _, _ = build_features(df, fit=False, pipeline_path=pipeline_path)
    threshold = 0.50
    best_params_path = MODEL_DIR / "best_params.json"
    if best_params_path.exists():
        with open(best_params_path) as f:
            threshold = float(json.load(f).get("threshold_cost", 0.50))

    # Score
    t0 = time.time()
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    latency = time.time() - t0

    # Save
    scored = df.copy()
    scored["churn_prediction"] = predictions
    scored["churn_probability"] = probabilities.round(4)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)

    print(f"  Scored {len(scored)} customers in {latency:.2f}s")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Predicted churners: {predictions.sum()} ({predictions.mean():.1%})")
    if TARGET in df.columns:
        actual = df[TARGET].sum()
        print(f"  Actual churners:    {actual} ({df[TARGET].mean():.1%})")
    print(f"  Output: {output_path}")
    print(f"  Avg latency per customer: {latency/len(df)*1000:.2f}ms")


if __name__ == "__main__":
    main()
