"""
prepare.py  -  Data Preparation
===============================
Describes the historical/current batch split used in the pipeline.
"""

import pandas as pd


def describe_split(
    train: pd.DataFrame,
    current: pd.DataFrame,
    current_stress: pd.DataFrame | None = None,
) -> dict:
    """Print and return statistics about the historical/current split."""
    stats = {
        "train_rows": len(train),
        "train_churn_rate": train["churn"].mean(),
        "current_rows": len(current),
        "current_churn_rate": current["churn"].mean(),
        "churn_delta": current["churn"].mean() - train["churn"].mean(),
    }
    if current_stress is not None:
        stats["current_stress_rows"] = len(current_stress)
        stats["current_stress_churn_rate"] = current_stress["churn"].mean()
        stats["stress_delta"] = current_stress["churn"].mean() - train["churn"].mean()

    print(f"\n{'=' * 60}")
    print("  DATA SPLIT - Historical vs Stable Current vs Stress Current")
    print(f"{'=' * 60}")
    print(f"  Historical batch: {stats['train_rows']:,} customers")
    print(f"    Churn rate: {stats['train_churn_rate']:.1%}")
    print(f"  Stable current:   {stats['current_rows']:,} customers")
    print(f"    Churn rate: {stats['current_churn_rate']:.1%}")
    print(f"  Churn-rate delta: {stats['churn_delta']:+.1%}")
    if current_stress is not None:
        print(f"  Stress current:   {stats['current_stress_rows']:,} customers")
        print(f"    Churn rate: {stats['current_stress_churn_rate']:.1%}")
        print(f"  Stress delta:     {stats['stress_delta']:+.1%}")
    print(f"{'=' * 60}\n")

    return stats
