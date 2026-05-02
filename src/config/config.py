"""
config.py  —  All constants in one place
=========================================
Module 2 counterpart: replaces hardcoded values scattered across scripts.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data"
MODEL_DIR   = ROOT / "models"
LOG_DIR     = ROOT / "logs"
MLRUNS_DIR  = ROOT / "mlruns"

FULL_DATA_CSV      = DATA_DIR / "telco_churn_full.csv"
TRAIN_CSV          = DATA_DIR / "train.csv"
CURRENT_CSV        = DATA_DIR / "current.csv"
CURRENT_STRESS_CSV = DATA_DIR / "current_stress.csv"

BATCH_PATHS = {
    "current": CURRENT_CSV,
    "stress": CURRENT_STRESS_CSV,
}

# ── Dataset ───────────────────────────────────────────────────────────────────
TARGET        = "churn"
ID_COL        = "customer_id"
TENURE_COL    = "tenure_months"
TENURE_CUTOFF = 12          # months - new customers (<=12) vs established (>12)
STRESS_BATCH_SIZE = 800
NORMAL_BATCH_SIZE = 943
STRESS_SCORE_QUOTAS = {
    4: 250,
    3: 250,
    2: 300,
}

NUMERIC_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges",
]
BINARY_FEATURES = [
    "gender", "senior_citizen", "partner", "dependents",
    "phone_service", "paperless_billing",
]
MULTI_FEATURES = [
    "multiple_lines", "internet_service", "online_security",
    "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies", "contract", "payment_method",
]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + MULTI_FEATURES

# ── Training ──────────────────────────────────────────────────────────────────
SEED            = 42
TEST_SIZE       = 0.20
LGBM_PARAMS     = dict(
    n_estimators    = 500,
    learning_rate   = 0.05,
    max_depth       = 6,
    num_leaves      = 31,
    min_child_samples = 20,
    subsample       = 0.8,
    colsample_bytree = 0.8,
    verbosity       = -1,
    random_state    = SEED,
)

# ── Deployment Gates ──────────────────────────────────────────────────────────
GATE_F1      = 0.48
GATE_RECALL  = 0.55
GATE_AUC_PR  = 0.40
GATE_AUC_ROC = 0.70
PRODUCTION_THRESHOLD = 0.20

# ── Business Cost Model ──────────────────────────────────────────────────────
COST_FN = 1500     # lost customer acquisition cost
COST_FP = 50       # retention offer wasted on non-churner
COST_RATIO = COST_FN / COST_FP   # 30:1

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT   = "telco-churn-pipeline"
MLFLOW_TRACKING_URI = f"file:///{MLRUNS_DIR.as_posix()}"
MLFLOW_MODEL_NAME   = "telco-churn-production"

# ── Drift Detection ──────────────────────────────────────────────────────────
DRIFT_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges",
    "contract", "internet_service", "payment_method",
]
DRIFT_THRESHOLD = 0.10   # Evidently column-level p-value threshold

# ── Retraining ────────────────────────────────────────────────────────────────
RETRAIN_COOLDOWN_DAYS = 14
