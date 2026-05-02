"""
main.py  --  Pipeline Entry Point
==================================
Runs the complete MLOps pipeline end-to-end.

    python main.py

THE SITUATION
-------------
A telecom company has 4,857 established customers (tenure > 12 months)
with a 17% churn rate. A model is trained, validated, and deployed.

After deployment, 2,186 new customers (tenure <= 12 months) arrive
with a 47% churn rate -- 2.8x higher. The model faces distribution
drift across tenure, contract type, internet service, and billing.

WHAT THIS PIPELINE DOES (12 stages)
------------------------------------
  Stage 0:  Initialisation (seed, directories, MLflow setup)
  Stage 1:  Data loading (train + current CSVs)
  Stage 2:  Data validation (Pandera schema)
  Stage 3:  Feature engineering (OHE, imputation, pipeline saved)
  Stage 4:  Train/test split (80/20, stratified)
  Stage 5:  Model training (LightGBM, MLflow tracked)
  Stage 6:  Pre-deployment evaluation (hold-out test set)
  Stage 7:  Deployment gates (F1 >= 0.55, Recall >= 0.55)
  Stage 8:  MLflow model logging (artefact + signature)
  Stage 9:  Model registry (versioning, staging, production)
  Stage 10: Drift detection (Evidently report)
  Stage 11: Batch scoring (new customer cohort)
  Stage 12: Lineage recording (data + model + code provenance)

WHAT CHANGED FROM MODULE 1
---------------------------
  experiments.csv          -> MLflow experiment tracking
  registry.json            -> MLflow Model Registry
  manual schema checks     -> Pandera data validation
  no data versioning       -> DVC dataset versioning
  manual drift detection   -> Evidently drift reports
  python main.py           -> GitHub Actions CI/CD (see .github/workflows/)

AFTER THIS PIPELINE
--------------------
  - View MLflow UI: mlflow ui --backend-store-uri file:///path/to/mlruns
  - View drift report: open logs/drift_report.html
  - Serve the model: uvicorn serve:app --reload --port 8000
  - Batch scoring: python batch_infer.py
  - Retrain on drift: python retrain.py --strategy mixed
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline.run_pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the telco churn MLOps pipeline.")
    parser.add_argument(
        "--batch",
        choices=["current", "stress"],
        default="current",
        help="Post-deployment batch to monitor and score (default: current).",
    )
    args = parser.parse_args()
    metrics = run_pipeline(monitor_batch=args.batch)
