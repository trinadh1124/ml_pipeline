"""
test_features.py  —  Unit tests for feature engineering
"""

import pandas as pd
import numpy as np
import pytest
from src.features.build_features import build_features


def _make_sample_df(n=20):
    """Create a small sample DataFrame for testing."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "senior_citizen": rng.choice(["Yes", "No"], n),
        "partner": rng.choice(["Yes", "No"], n),
        "dependents": rng.choice(["Yes", "No"], n),
        "tenure_months": rng.randint(1, 72, n),
        "phone_service": rng.choice(["Yes", "No"], n),
        "multiple_lines": rng.choice(["Yes", "No", "No phone service"], n),
        "internet_service": rng.choice(["DSL", "Fiber optic", "No"], n),
        "online_security": rng.choice(["Yes", "No", "No internet service"], n),
        "online_backup": rng.choice(["Yes", "No", "No internet service"], n),
        "device_protection": rng.choice(["Yes", "No", "No internet service"], n),
        "tech_support": rng.choice(["Yes", "No", "No internet service"], n),
        "streaming_tv": rng.choice(["Yes", "No", "No internet service"], n),
        "streaming_movies": rng.choice(["Yes", "No", "No internet service"], n),
        "contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "paperless_billing": rng.choice(["Yes", "No"], n),
        "payment_method": rng.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ], n),
        "monthly_charges": rng.uniform(20, 100, n).round(2),
        "total_charges": rng.uniform(100, 5000, n).round(2),
        "churn": rng.choice([0, 1], n, p=[0.7, 0.3]),
    })


def test_output_shape():
    """Feature engineering should produce correct number of rows."""
    df = _make_sample_df(20)
    X, names, info = build_features(df, fit=True)
    assert len(X) == 20
    assert len(names) > 0


def test_no_target_in_features():
    """Target column should not appear in features."""
    df = _make_sample_df(20)
    X, names, _ = build_features(df, fit=True)
    assert "churn" not in names


def test_no_id_in_features():
    """Customer ID should not appear in features."""
    df = _make_sample_df(20)
    X, names, _ = build_features(df, fit=True)
    assert "customer_id" not in names


def test_binary_encoding():
    """Binary features should be 0 or 1."""
    df = _make_sample_df(50)
    X, names, _ = build_features(df, fit=True)
    for col in ["gender", "senior_citizen", "partner", "dependents"]:
        if col in X.columns:
            assert set(X[col].unique()).issubset({0, 1})
