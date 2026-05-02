"""
versioning.py  —  MLflow Model Registry
=========================================
MODULE 2 KEY TOOL: MLflow Model Registry replaces the JSON registry from Module 1.

In Module 1:
    registry = json.load(open("models/registry.json"))
    registry["production"] = {"model_path": "...", "f1": 0.72}

Now the MLflow Model Registry handles:
    - Model versioning (v1, v2, v3...)
    - Stage transitions (None -> Staging -> Production)
    - Descriptions and tags
    - Programmatic model loading by stage
"""

import mlflow
from mlflow.tracking import MlflowClient
from src.config.config import MLFLOW_MODEL_NAME, MLFLOW_TRACKING_URI


def get_client() -> MlflowClient:
    """Get an MLflow client."""
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def register_model(run_id: str, model_name: str = MLFLOW_MODEL_NAME) -> str:
    """
    Register a logged model in the MLflow Model Registry.
    Returns the model version string.
    """
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    version = result.version
    print(f"  [registry] Registered {model_name} version {version}")
    return version


def promote_to_staging(version: str, model_name: str = MLFLOW_MODEL_NAME):
    """Tag a model version as 'staging'."""
    client = get_client()
    client.set_model_version_tag(model_name, version, "stage", "staging")
    print(f"  [registry] {model_name} v{version} -> Staging")


def promote_to_production(version: str, model_name: str = MLFLOW_MODEL_NAME):
    """Tag a model version as 'production'."""
    client = get_client()
    # Mark all previous versions as archived
    for mv in client.search_model_versions(f"name='{model_name}'"):
        tags = mv.tags or {}
        if tags.get("stage") == "production" and mv.version != version:
            client.set_model_version_tag(model_name, mv.version, "stage", "archived")
    client.set_model_version_tag(model_name, version, "stage", "production")
    print(f"  [registry] {model_name} v{version} -> Production")


def load_production_model(model_name: str = MLFLOW_MODEL_NAME):
    """Load the latest production model from the registry."""
    client = get_client()
    versions = client.search_model_versions(f"name='{model_name}'")
    prod_versions = [v for v in versions if (v.tags or {}).get("stage") == "production"]
    if not prod_versions:
        raise RuntimeError(f"No production model found for '{model_name}'")
    latest = sorted(prod_versions, key=lambda v: int(v.version))[-1]
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest.version}")
    print(f"  [registry] Loaded {model_name} v{latest.version} (production)")
    return model


def get_model_info(model_name: str = MLFLOW_MODEL_NAME) -> list[dict]:
    """Get info about all registered model versions."""
    client = get_client()
    versions = client.search_model_versions(f"name='{model_name}'")
    return [
        {
            "version": v.version,
            "stage": (v.tags or {}).get("stage", "none"),
            "run_id": v.run_id,
            "created": v.creation_timestamp,
        }
        for v in versions
    ]
