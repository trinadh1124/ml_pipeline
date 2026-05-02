# MLOps Pipeline Module 2 - Telecom Churn Prediction

From manual to production tools. Every concept built from scratch in Module 1
is now replaced with its production counterpart.

---

## The Situation

A telecom company is losing 26.5% of its customers. Each lost customer costs
$1,500 in acquisition to replace. The data science team has data on 7,043
customers - demographics, services, contract details, and churn status.

The corrected production story now uses **one historical batch plus two
post-deployment batches** built from the same original 7,043 customers:

- `train.csv` - historical baseline available at deployment (`5,300` rows)
- `current.csv` - normal incoming production batch (`943` rows)
- `current_stress.csv` - intentionally shifted post-deployment batch (`800` rows)

This gives us both narratives we want:
- a stable production batch where monitoring stays calm
- a separate stress batch where drift is visible and retraining is justified

---

## What Changed from Module 1

| Module 1 (from scratch) | Module 2 (production tool) |
|---|---|
| `experiments.csv` | **MLflow** experiment tracking |
| `registry.json` | **MLflow Model Registry** |
| Manual `if`-`else` checks | **Pandera** data validation |
| No data versioning | **DVC** dataset versioning + pipeline |
| Manual mean/std drift | **Evidently** drift reports |
| `python main.py` | **GitHub Actions** CI/CD |
| `serve.py` only | **Docker + FastAPI** containerised |
| `feature_importances_` | **SHAP** per-prediction explanations |
| Fixed hyperparameters | **Optuna** Bayesian tuning |
| No tests | **pytest** (29 tests, unit + integration) |

---

## Quick Start

```bash
# 1. Unzip and enter the folder
cd MLOps_Cloud_Pipeline

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Data is included in the zip
#    data/train.csv, data/current.csv, data/current_stress.csv
#    No download needed. To verify:
python -c "import pandas as pd; print(pd.read_csv('data/train.csv').shape)"
# Expected: (5300, 21)

# 5. Initial deployment flow
python baseline.py
python model_selection.py
python main.py
python main.py --batch stress

# 6. Serve / score after deployment
uvicorn serve:app --reload --port 8000
python batch_infer.py
python batch_infer.py --batch stress

# 7. Post-deployment drift response
python retrain.py --dry-run --batch current
python retrain.py --dry-run --batch stress
python retrain.py --strategy mixed

# 8. View MLflow UI (run after main.py)
mlflow ui --backend-store-uri ./mlruns

# 9. View Evidently drift report (run after main.py)
# Open logs/drift_report.html in any browser

# 10. Run all tests
pytest tests/ -v
```

---

## Pipeline Stages (main.py)

| Stage | What it does | Tool |
|---|---|---|
| 0 | Initialisation (seed, dirs, MLflow) | MLflow |
| 1 | Data loading (historical + selectable post-deployment batch) | DVC |
| 2 | Data validation | **Pandera** |
| 3 | Feature engineering (OHE, imputation) | - |
| 4 | Train/test split (80/20, stratified) | - |
| 5 | Model training (winner from model selection, currently XGBoost) | **MLflow** |
| 6 | Pre-deployment evaluation | - |
| 7 | Deployment gates | - |
| 8 | MLflow model logging | **MLflow** |
| 9 | Model registry promotion | **MLflow** |
| 10 | Drift detection against `current` or `current_stress` | **Evidently** |
| 11 | Batch scoring for the selected post-deployment batch | - |
| 12 | Lineage recording | - |

---

## Retraining Strategies

| Strategy | When to use |
|---|---|
| `finetune` | Mild drift, tight compute budget |
| `mixed` | Need one model that works across historical, stable current, and stress traffic |
| `reexperiment` | Metrics degrade enough to justify a fresh search |

Recommended default: `mixed` for the stress batch, because it adapts to the shifted segment without throwing away the historical baseline.

The project now uses a single fixed production threshold of `0.20` everywhere: hold-out evaluation, batch scoring, and retraining evaluation.

---

## DVC Setup

```bash
dvc init
dvc remote add -d local_remote ./dvc_remote
dvc add data/train.csv data/current.csv data/current_stress.csv
git add data/*.dvc
dvc push
```

### DVC Pipeline (dvc.yaml)

```bash
dvc repro
dvc repro train
dvc dag
dvc metrics show
dvc metrics diff
```

---

## Docker

```bash
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
curl http://localhost:8000/health
```

---

## Tests

```bash
pytest tests/ -v
```

29 tests across 5 modules:
- `test_validate.py` - 6 Pandera schema tests
- `test_features.py` - 4 feature engineering tests
- `test_train.py` - 5 model training tests
- `test_evaluate.py` - 7 evaluation and gate tests
- `test_pipeline_e2e.py` - 7 integration tests

---

## Expected Results

**Final production model:** tuned `LGBMClassifier`

**Pre-deployment hold-out (default threshold 0.50):**
- F1: `0.571`
- Recall: `0.656`
- Precision: `0.505`
- AUC-ROC: `0.841`
- AUC-PR: `0.533`

**Production threshold used by serving/batch (`0.20`):**
- Hold-out metrics and post-deployment scoring now use the same fixed learner-facing threshold everywhere

**Monitoring story:**
- `current.csv` is the stable post-deployment batch used to show "no retrain needed"
- `current_stress.csv` is the shifted campaign-style batch used to show drift and a retraining trigger
- `mixed` is the recommended retraining response for the stress batch
- the same `0.20` threshold is used consistently across deployment and retraining
- `main.py --batch stress` is monitor-only, so it does not replace the stable production model

---

## File Reference

| File | What it does | Tool |
|---|---|---|
| `baseline.py` | Naive approach - 7 pitfalls | - |
| `model_selection.py` | 4-model comparison + Optuna tuning | MLflow |
| `main.py` | Full 12-stage pipeline | All tools |
| `retrain.py` | Drift-triggered retraining | Evidently, MLflow |
| `serve.py` | FastAPI REST endpoint | FastAPI |
| `batch_infer.py` | Batch scorer | MLflow |
| `plot_logs.py` | HTML dashboard (MLflow + drift + lineage) | MLflow |
| `Dockerfile` | Container definition | Docker |
| `.github/workflows/` | CI/CD pipeline | GitHub Actions |
| `dvc.yaml` | Reproducible pipeline stages | DVC |
| `src/models/explain.py` | SHAP global + per-customer explanations | SHAP |
| `src/models/tuning.py` | Bayesian hyperparameter search | Optuna |
| `tests/` | 29 tests (unit + integration) | pytest |

---

**[GUIDE.html](GUIDE.html)** - open in Chrome/Edge for the complete walkthrough.
