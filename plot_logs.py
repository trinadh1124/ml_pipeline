"""
plot_logs.py  —  MLflow Run Visualizer
========================================
Opens an HTML dashboard showing all MLflow experiment runs,
model registry status, and drift detection results.

In Module 1 this read from CSV/JSON logs.
In Module 2 it reads directly from MLflow.

    python plot_logs.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import webbrowser
from pathlib import Path
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
from src.config.config import MLFLOW_TRACKING_URI, MLFLOW_MODEL_NAME, LOG_DIR

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# ── Gather data ───────────────────────────────────────────────────────────────

# All experiments and runs
experiments = []
for exp in client.search_experiments():
    runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"])
    for r in runs:
        experiments.append({
            "experiment": exp.name,
            "run_name": r.data.tags.get("mlflow.runName", "unnamed"),
            "run_id": r.info.run_id[:12],
            "f1": r.data.metrics.get("f1", 0),
            "recall": r.data.metrics.get("recall", 0),
            "precision": r.data.metrics.get("precision", 0),
            "auc_roc": r.data.metrics.get("auc_roc", 0),
            "missed": int(r.data.metrics.get("missed_churners", 0)),
            "cost": int(r.data.metrics.get("total_business_cost", 0)),
            "model": r.data.params.get("model", "LightGBM"),
            "time": datetime.fromtimestamp(r.info.start_time / 1000).strftime("%Y-%m-%d %H:%M"),
        })

# Model registry
registry = []
try:
    versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
    for v in versions:
        tags = v.tags or {}
        registry.append({
            "version": v.version,
            "stage": tags.get("stage", "none"),
            "run_id": v.run_id[:12],
            "created": datetime.fromtimestamp(v.creation_timestamp / 1000).strftime("%Y-%m-%d %H:%M"),
        })
except Exception:
    pass

# Lineage
lineage = {}
lineage_path = LOG_DIR / "lineage.json"
if lineage_path.exists():
    with open(lineage_path) as f:
        lineage = json.load(f)

# Drift reports
drift_current_path = LOG_DIR / "drift_report_current.html"
drift_stress_path = LOG_DIR / "drift_report_stress.html"
drift_current_exists = drift_current_path.exists()
drift_stress_exists = drift_stress_path.exists()

# ── Build HTML ────────────────────────────────────────────────────────────────

def make_row(d, keys):
    cells = ""
    for k in keys:
        v = d.get(k, "")
        if isinstance(v, float):
            v = f"{v:.3f}"
        elif isinstance(v, int) and k == "cost":
            v = f"${v:,}"
        cells += f"<td>{v}</td>"
    return f"<tr>{cells}</tr>"

exp_rows = "\n".join(
    make_row(e, ["time", "experiment", "run_name", "model", "f1", "recall", "precision", "auc_roc", "missed", "cost"])
    for e in experiments
)

reg_rows = "\n".join(
    make_row(r, ["version", "stage", "run_id", "created"])
    for r in registry
)

drift_links = []
if drift_current_exists:
    drift_links.append(
        f'<a href="file:///{drift_current_path.as_posix()}" target="_blank">Open Stable Batch Drift Report</a>'
    )
if drift_stress_exists:
    drift_links.append(
        f'<a href="file:///{drift_stress_path.as_posix()}" target="_blank">Open Stress Batch Drift Report</a>'
    )
drift_link = " | ".join(drift_links) if drift_links else "Not generated yet"

lineage_info = ""
if lineage:
    m = lineage.get("metrics", {})
    lineage_info = f"""
    <tr><td>MLflow Run ID</td><td>{lineage.get('mlflow_run_id', 'N/A')[:12]}</td></tr>
    <tr><td>Train data</td><td>{lineage.get('train_checksum', 'N/A')}</td></tr>
    <tr><td>Current data</td><td>{lineage.get('current_checksum', 'N/A')}</td></tr>
    <tr><td>Python</td><td>{lineage.get('python_version', 'N/A')[:20]}</td></tr>
    <tr><td>Pipeline version</td><td>{lineage.get('pipeline_version', 'N/A')}</td></tr>
    <tr><td>F1 (pre-deploy)</td><td>{m.get('f1', 0):.3f}</td></tr>
    <tr><td>Business cost</td><td>${m.get('total_business_cost', 0):,}</td></tr>
    """

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MLOps Dashboard - Telco Churn</title>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d27; --surface2: #22263a;
    --border: #2e3350; --accent: #5b8dee; --accent2: #3ecf8e;
    --warn: #f59e0b; --danger: #ef4444; --text: #e2e8f0; --muted: #8892a4;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); padding: 40px;
  }}
  h1 {{ font-size: 24px; margin-bottom: 8px; }}
  .sub {{ color: var(--accent); font-size: 13px; text-transform: uppercase;
          letter-spacing: 0.08em; margin-bottom: 32px; }}
  h2 {{
    font-size: 16px; color: #fff; margin: 32px 0 12px;
    padding-bottom: 8px; border-bottom: 1px solid var(--border);
  }}
  table {{
    width: 100%; border-collapse: collapse; margin: 12px 0 24px; font-size: 13px;
  }}
  th {{
    background: var(--surface2); color: var(--text); text-align: left;
    padding: 8px 12px; border: 1px solid var(--border);
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
  }}
  td {{
    padding: 7px 12px; border: 1px solid var(--border); color: var(--muted);
  }}
  tr:hover td {{ background: var(--surface2); }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px 20px; margin: 12px 0;
  }}
  a {{ color: var(--accent); }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 600; text-transform: uppercase;
  }}
  .badge.green {{ background: rgba(62,207,142,0.15); color: var(--accent2); }}
  .badge.blue {{ background: rgba(91,141,238,0.15); color: var(--accent); }}
</style>
</head>
<body>
<h1>MLOps Dashboard</h1>
<div class="sub">Telco Churn Pipeline &mdash; Module 2</div>

<h2>Experiment Runs (MLflow)</h2>
<table>
  <tr>
    <th>Time</th><th>Experiment</th><th>Run</th><th>Model</th>
    <th>F1</th><th>Recall</th><th>Precision</th><th>AUC-ROC</th>
    <th>Missed</th><th>Cost</th>
  </tr>
  {exp_rows}
</table>

<h2>Model Registry</h2>
<table>
  <tr><th>Version</th><th>Stage</th><th>Run ID</th><th>Created</th></tr>
  {reg_rows}
</table>

<h2>Drift Detection</h2>
<div class="card">
  <p>{drift_link}</p>
  <p style="color: var(--muted); font-size: 13px; margin-top: 8px;">
    Stable monitoring should stay calm on <code>current.csv</code>, while
    <code>current_stress.csv</code> is the learner-facing batch that demonstrates
    visible drift and retraining.
  </p>
</div>

<h2>Lineage (Latest Pipeline Run)</h2>
<table>
  <tr><th>Field</th><th>Value</th></tr>
  {lineage_info}
</table>

<p style="color: var(--muted); font-size: 12px; margin-top: 40px;">
  For the full interactive MLflow UI, run:
  <code style="background: var(--surface2); padding: 2px 6px; border-radius: 3px;">
    mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}
  </code>
</p>
</body>
</html>"""

# Write and open
out = LOG_DIR / "dashboard.html"
LOG_DIR.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Dashboard saved to {out}")
webbrowser.open(f"file:///{out.as_posix()}")
