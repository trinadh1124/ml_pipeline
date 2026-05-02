"""
lineage.py  —  Provenance & Lineage
=====================================
Records what data + code + environment produced each model.
Now linked to MLflow run IDs for complete traceability.
"""

import json
import hashlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from src.config.config import LOG_DIR


def compute_checksum(filepath: str | Path) -> str:
    """SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def record_lineage(
    model_path: str,
    train_path: str,
    current_path: str,
    mlflow_run_id: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Record full provenance snapshot."""
    lineage = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_file": str(model_path),
        "train_data": str(train_path),
        "train_checksum": compute_checksum(train_path),
        "current_data": str(current_path),
        "current_checksum": compute_checksum(current_path),
        "python_version": sys.version,
        "platform": platform.platform(),
        "mlflow_run_id": mlflow_run_id,
    }
    if extra:
        lineage.update(extra)

    # Also try to get git commit
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lineage["git_commit"] = result.stdout.strip()
    except Exception:
        lineage["git_commit"] = "not-a-git-repo"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / "lineage.json"
    with open(out, "w") as f:
        json.dump(lineage, f, indent=2)
    print(f"  [lineage] Provenance recorded -> {out}")
    return lineage
