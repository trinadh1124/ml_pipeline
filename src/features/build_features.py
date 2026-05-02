"""
build_features.py  --  Feature Engineering
==========================================
Transforms raw Telco churn data into model-ready features.

Beyond basic encoding, this adds domain-driven features that directly
reflect signals found in EDA:
  - Month-to-month contracts have 11x higher churn than two-year
  - Fiber optic users churn at 3x the rate of DSL users
  - Service count has a U-shape: 1-2 services = peak churn risk
  - Tenure Q1 (new customers) churn at 4x the rate of loyal customers
  - High charges + fiber optic is the highest-risk combination

Saves the feature pipeline artefact for inference-time consistency.
Also saves data/features_train.csv as a DVC pipeline stage output.
"""

import json
import pandas as pd
import numpy as np
from src.config.config import (
    NUMERIC_FEATURES, BINARY_FEATURES, MULTI_FEATURES, TARGET, ID_COL,
    MODEL_DIR, DATA_DIR,
)

SERVICE_COLS = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]


def _add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven features before encoding."""

    # Contract risk: Month-to-month=2, One year=1, Two year=0
    contract_risk = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    df["contract_risk"] = df["contract"].map(contract_risk).fillna(1).astype(int)
    df["is_month_to_month"] = (df["contract"] == "Month-to-month").astype(int)

    # Internet type: Fiber 30% churn vs DSL 9.5% vs No 2%
    df["is_fiber"] = (df["internet_service"] == "Fiber optic").astype(int)
    df["no_internet"] = (df["internet_service"] == "No").astype(int)

    # Service count U-shape: 1-2 services = peak churn, 6 services = lowest
    def count_yes(row):
        return sum(1 for c in SERVICE_COLS if c in row.index and row[c] == "Yes")

    df["service_count"] = df.apply(count_yes, axis=1)
    df["service_count_sq"] = df["service_count"] ** 2

    protection_cols = ["online_security", "online_backup", "device_protection", "tech_support"]
    df["protection_count"] = df[[c for c in protection_cols if c in df.columns]].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )
    df["has_protection_bundle"] = (df["protection_count"] >= 2).astype(int)

    if "streaming_tv" in df.columns and "streaming_movies" in df.columns:
        df["has_both_streaming"] = (
            (df["streaming_tv"] == "Yes") & (df["streaming_movies"] == "Yes")
        ).astype(int)

    # Tenure bins: New (0-12) 27.5% churn, Mid (13-36), Loyal (37+) 6.4%
    df["is_new_customer"] = (df["tenure_months"] <= 12).astype(int)
    df["is_loyal_customer"] = (df["tenure_months"] >= 37).astype(int)
    df["tenure_bin"] = pd.cut(
        df["tenure_months"], bins=[-1, 12, 36, 999], labels=[0, 1, 2],
    ).astype(int)

    # High-risk combinations
    monthly_median = df["monthly_charges"].median()
    df["is_high_charge"] = (df["monthly_charges"] > monthly_median).astype(int)
    df["fiber_month_to_month"] = (df["is_fiber"] & df["is_month_to_month"]).astype(int)
    df["high_charge_fiber"] = (df["is_high_charge"] & df["is_fiber"]).astype(int)
    df["highest_risk"] = (
        df["is_month_to_month"] & df["is_fiber"] & df["is_high_charge"]
    ).astype(int)

    # Payment method: electronic check = highest churn, auto-pay = lowest
    df["is_electronic_check"] = (df["payment_method"] == "Electronic check").astype(int)
    df["is_auto_pay"] = df["payment_method"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)
    df["is_mailed_check"] = (df["payment_method"] == "Mailed check").astype(int)

    # Demographic vulnerability: senior with no support network
    if "senior_citizen" in df.columns:
        senior_flag = df["senior_citizen"].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0)
        no_partner = (df["partner"] == "No").astype(int) if "partner" in df.columns else 0
        no_dep = (df["dependents"] == "No").astype(int) if "dependents" in df.columns else 0
        df["senior_alone"] = (senior_flag & no_partner & no_dep).astype(int)

    # Charge ratios
    df["charge_per_tenure"] = df["monthly_charges"] / (df["tenure_months"] + 1)
    df["total_vs_expected"] = df["total_charges"] / (
        df["monthly_charges"] * df["tenure_months"] + 1
    )
    df["charge_per_service"] = df["monthly_charges"] / (df["service_count"] + 1)
    df["service_density"] = df["service_count"] / (df["tenure_months"] + 1)
    df["charges_x_tenure"] = df["monthly_charges"] * np.log1p(df["tenure_months"])

    # Additional domain interactions for churn-heavy segments
    df["month_to_month_echeck"] = (
        df["is_month_to_month"] & df["is_electronic_check"]
    ).astype(int)
    df["fiber_echeck"] = (
        df["is_fiber"] & df["is_electronic_check"]
    ).astype(int)
    df["new_fiber_customer"] = (
        df["is_new_customer"] & df["is_fiber"]
    ).astype(int)
    df["new_month_to_month"] = (
        df["is_new_customer"] & df["is_month_to_month"]
    ).astype(int)
    df["high_charge_monthly"] = (
        df["is_high_charge"] & df["is_month_to_month"]
    ).astype(int)
    df["monthly_no_protection"] = (
        df["is_month_to_month"] & (df["has_protection_bundle"] == 0)
    ).astype(int)
    df["fiber_no_support"] = (
        df["is_fiber"] & (df["tech_support"] == "No")
    ).astype(int)
    df["loyal_auto_pay"] = (
        df["is_loyal_customer"] & df["is_auto_pay"]
    ).astype(int)
    df["streaming_fiber"] = (
        df["is_fiber"] & df.get("has_both_streaming", 0)
    ).astype(int)

    return df


def build_features(
    df: pd.DataFrame,
    fit: bool = True,
    pipeline_path: str | None = None,
    save_features_csv: bool = False,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Transform raw data into model-ready features.

    Args:
        df:               Raw DataFrame.
        fit:              True = compute medians/cols and save pipeline.
                          False = load saved pipeline for inference consistency.
        pipeline_path:    Path to feature_pipeline.json.
        save_features_csv: Save engineered matrix to data/features_train.csv
                          for DVC versioning.

    Returns:
        (X, feature_names, pipeline_info)
    """
    df = df.copy()

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    df = _add_domain_features(df)

    # Impute numeric
    all_numeric = NUMERIC_FEATURES + [
        "charge_per_tenure", "total_vs_expected", "charge_per_service",
        "service_count", "service_count_sq", "protection_count",
        "contract_risk", "tenure_bin", "service_density", "charges_x_tenure",
    ]
    if fit:
        medians = {
            col: float(df[col].median())
            for col in all_numeric
            if col in df.columns and df[col].isnull().any()
        }
        if "total_charges" in df.columns and df["total_charges"].isnull().any():
            medians["total_charges"] = float(df["total_charges"].median())
    else:
        with open(pipeline_path) as f:
            saved = json.load(f)
        medians = saved["medians"]

    for col, val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Encode binary
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in BINARY_FEATURES:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0).astype(int)

    # One-hot encode
    df = pd.get_dummies(df, columns=MULTI_FEATURES, drop_first=True, dtype=int)

    if fit:
        feature_cols = [c for c in df.columns if c != TARGET]
        pipeline_info = {
            "medians":      medians,
            "feature_cols": feature_cols,
            "binary_map":   binary_map,
        }
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        save_path = pipeline_path or str(MODEL_DIR / "feature_pipeline.json")
        with open(save_path, "w") as f:
            json.dump(pipeline_info, f, indent=2)
        print(f"  [features] Saved feature pipeline -> {save_path}")
        print(f"  [features] {len(feature_cols)} features "
              f"(+{len(feature_cols) - len(NUMERIC_FEATURES + BINARY_FEATURES)} vs raw encoding)")

        if save_features_csv:
            out_path = DATA_DIR / "features_train.csv"
            cols_to_save = feature_cols + ([TARGET] if TARGET in df.columns else [])
            df[[c for c in cols_to_save if c in df.columns]].to_csv(out_path, index=False)
            print(f"  [features] Saved features_train.csv -> {out_path} (DVC-tracked)")
    else:
        with open(pipeline_path) as f:
            pipeline_info = json.load(f)
        feature_cols = pipeline_info["feature_cols"]
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[[c for c in feature_cols if c in df.columns]]

    X = df[[c for c in feature_cols if c in df.columns]]
    return X, list(X.columns), pipeline_info
