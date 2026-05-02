"""
loader.py  -  Dataset Loading and Split Creation
================================================
Creates three deterministic datasets from the original 7,043-row Telco file:
  - train.csv          : historical data available at deployment
  - current.csv        : normal post-deployment arrivals
  - current_stress.csv : intentionally shifted post-deployment batch
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.config import (
    BATCH_PATHS,
    CURRENT_CSV,
    CURRENT_STRESS_CSV,
    DATA_DIR,
    FULL_DATA_CSV,
    NORMAL_BATCH_SIZE,
    SEED,
    STRESS_BATCH_SIZE,
    STRESS_SCORE_QUOTAS,
    TENURE_CUTOFF,
    TRAIN_CSV,
)


RAW_TO_CANONICAL = {
    "CustomerID": "customer_id",
    "Gender": "gender",
    "Senior Citizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "Tenure Months": "tenure_months",
    "Phone Service": "phone_service",
    "Multiple Lines": "multiple_lines",
    "Internet Service": "internet_service",
    "Online Security": "online_security",
    "Online Backup": "online_backup",
    "Device Protection": "device_protection",
    "Tech Support": "tech_support",
    "Streaming TV": "streaming_tv",
    "Streaming Movies": "streaming_movies",
    "Contract": "contract",
    "Paperless Billing": "paperless_billing",
    "Payment Method": "payment_method",
    "Monthly Charges": "monthly_charges",
    "Total Charges": "total_charges",
    "Churn Value": "churn",
}

CANONICAL_COLUMNS = list(RAW_TO_CANONICAL.values())


def load_train() -> pd.DataFrame:
    """Load the historical batch available at deployment time."""
    ensure_split_files()
    return _load_csv(TRAIN_CSV)


def load_current(batch_name: str = "current") -> pd.DataFrame:
    """Load a named post-deployment monitoring batch."""
    ensure_split_files()
    return _load_csv(resolve_batch_path(batch_name))


def load_current_stress() -> pd.DataFrame:
    """Load the intentionally shifted post-deployment batch."""
    return load_current(batch_name="stress")


def ensure_split_files(force: bool = False) -> None:
    """Create train/current/current_stress CSVs if they are missing."""
    required = [TRAIN_CSV, CURRENT_CSV, CURRENT_STRESS_CSV]
    if force or any(not path.exists() for path in required):
        _download_and_split(force=force)


def resolve_batch_path(batch_name: str) -> os.PathLike:
    """Return the CSV path for a named post-deployment batch."""
    try:
        return BATCH_PATHS[batch_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown batch '{batch_name}'. Use one of: {', '.join(BATCH_PATHS)}"
        ) from exc


def summarize_named_batch(batch_name: str) -> dict:
    """Return quick stats for a post-deployment batch."""
    df = load_current(batch_name=batch_name)
    return {
        "batch": batch_name,
        "rows": len(df),
        "churn_rate": float(df["churn"].mean()),
    }


def _load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["tenure_months"] = pd.to_numeric(df["tenure_months"], errors="coerce")
    return df


def _download_and_split(force: bool = False) -> None:
    """Download from Kaggle if needed and create the three-way split."""
    if not FULL_DATA_CSV.exists():
        _download_full_dataset()

    df = _load_raw_or_canonical(FULL_DATA_CSV)
    train, current, current_stress = _create_monitoring_splits(df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False)
    current.to_csv(CURRENT_CSV, index=False)
    current_stress.to_csv(CURRENT_STRESS_CSV, index=False)

    print(
        "[loader] Saved "
        f"{len(train)} train rows, "
        f"{len(current)} current rows, "
        f"{len(current_stress)} current_stress rows"
    )


def _download_full_dataset() -> None:
    import kagglehub

    path = Path(kagglehub.dataset_download("yeanzc/telco-customer-churn-ibm-dataset"))
    xlsx = next((path / f).as_posix() for f in os.listdir(path) if f.endswith(".xlsx"))
    raw = pd.read_excel(xlsx)
    canonical = _canonicalize_dataframe(raw)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(FULL_DATA_CSV, index=False)
    print(f"[loader] Downloaded full dataset -> {FULL_DATA_CSV}")


def _load_raw_or_canonical(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _canonicalize_dataframe(df)


def _canonicalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    canonical = df.rename(columns=RAW_TO_CANONICAL).copy()
    missing = [col for col in CANONICAL_COLUMNS if col not in canonical.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    canonical = canonical[CANONICAL_COLUMNS].copy()
    canonical["total_charges"] = pd.to_numeric(canonical["total_charges"], errors="coerce")
    canonical["tenure_months"] = pd.to_numeric(canonical["tenure_months"], errors="coerce")
    canonical["monthly_charges"] = pd.to_numeric(canonical["monthly_charges"], errors="coerce")
    canonical["churn"] = pd.to_numeric(canonical["churn"], errors="coerce").astype(int)
    return canonical


def _create_monitoring_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build deterministic historical, normal-current, and stress-current splits."""
    ranked = _rank_stress_candidates(df)
    stress = _select_stress_batch(ranked)
    remaining = ranked.drop(index=stress.index).drop(columns=["_stress_score"]).reset_index(drop=True)
    stress = stress.drop(columns=["_stress_score"]).reset_index(drop=True)

    stratify_key = _build_stratify_key(remaining)
    train, current = train_test_split(
        remaining,
        test_size=NORMAL_BATCH_SIZE,
        random_state=SEED,
        stratify=stratify_key,
    )

    return (
        train.reset_index(drop=True),
        current.reset_index(drop=True),
        stress,
    )


def _rank_stress_candidates(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["_stress_score"] = (
        (ranked["contract"] == "Month-to-month").astype(int)
        + (ranked["internet_service"] == "Fiber optic").astype(int)
        + (ranked["payment_method"] == "Electronic check").astype(int)
        + (ranked["tenure_months"] <= TENURE_CUTOFF).astype(int)
    )
    ranked = ranked.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    ranked = ranked.sort_values(
        by=["_stress_score", "monthly_charges", "tenure_months"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return ranked


def _select_stress_batch(ranked: pd.DataFrame) -> pd.DataFrame:
    """Sample a shifted but still credible stress batch from score buckets."""
    selected_parts: list[pd.DataFrame] = []
    taken_idx: set[int] = set()

    for score, target_rows in STRESS_SCORE_QUOTAS.items():
        bucket = ranked[ranked["_stress_score"] == score]
        chosen = bucket.head(target_rows)
        selected_parts.append(chosen)
        taken_idx.update(chosen.index.tolist())

    selected = pd.concat(selected_parts, axis=0)
    shortfall = STRESS_BATCH_SIZE - len(selected)
    if shortfall > 0:
        backfill = ranked.loc[~ranked.index.isin(taken_idx)].head(shortfall)
        selected = pd.concat([selected, backfill], axis=0)

    return selected.sort_values(
        by=["_stress_score", "monthly_charges", "tenure_months"],
        ascending=[False, False, True],
        kind="mergesort",
    )


def _build_stratify_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["churn"].astype(str)
        + "__"
        + df["contract"].astype(str)
        + "__"
        + df["internet_service"].astype(str)
    )
