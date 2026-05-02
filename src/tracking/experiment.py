"""
experiment.py  —  MLflow Experiment Tracking
=============================================
MODULE 2 KEY TOOL: MLflow replaces the CSV-based experiment log from Module 1.

In Module 1:
    with open("logs/experiments.csv", "a") as f:
        f.write(f"{timestamp},{params},{metrics}\\n")

Now MLflow does this with:
    - Automatic parameter/metric logging
    - Artefact storage (model files, plots, feature pipeline)
    - UI for comparing runs
    - Programmatic access to run history
"""

import mlflow
from mlflow.models import infer_signature
from src.config.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT


def setup_mlflow():
    """Configure MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"  [mlflow] Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"  [mlflow] Experiment:   {MLFLOW_EXPERIMENT}")


def start_run(run_name: str = None):
    """Start an MLflow run. Returns the run context."""
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict):
    """Log parameters to the active MLflow run."""
    mlflow.log_params(params)


def log_metrics(metrics: dict):
    """Log metrics to the active MLflow run."""
    mlflow.log_metrics(metrics)


def log_model(model, X_sample, model_name: str = "model"):
    """Log a scikit-learn compatible model to MLflow."""
    signature = infer_signature(X_sample, model.predict(X_sample))
    mlflow.sklearn.log_model(
        model, model_name, signature=signature,
    )
    print(f"  [mlflow] Model logged as artefact '{model_name}'")


def log_artifact(path: str):
    """Log a file as an artefact."""
    mlflow.log_artifact(path)


def get_run_id() -> str:
    """Get the current active run ID."""
    return mlflow.active_run().info.run_id
