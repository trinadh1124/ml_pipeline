"""
conftest.py  —  Shared test fixtures
======================================
Reusable data generators and fixtures for all test modules.
"""

import numpy as np
import pandas as pd
import pytest


def make_sample_df(n=50):
    """Create a synthetic telco customer DataFrame for testing."""
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
