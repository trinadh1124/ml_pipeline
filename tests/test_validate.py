"""
test_validate.py  —  Unit tests for Pandera data validation
=============================================================
Tests the four ML pipeline test types mentioned in the design doc:
  1. Unit tests (this file)     — individual function correctness
  2. Integration tests          — end-to-end pipeline runs
  3. Regression tests           — model metrics don't degrade
  4. Load tests                 — latency under batch scoring
"""

import pandas as pd
import pytest
import pandera as pa
from src.data.validate import validate_dataframe, telco_schema


def _make_valid_row(**overrides):
    """Create a single valid customer row."""
    row = {
        "customer_id": "TEST-001",
        "gender": "Male",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 24,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 70.35,
        "total_charges": 1397.47,
        "churn": 0,
    }
    row.update(overrides)
    return row


def test_valid_row_passes():
    """A well-formed row should pass validation."""
    df = pd.DataFrame([_make_valid_row()])
    result = validate_dataframe(df, "test")
    assert len(result) == 1


def test_invalid_gender_fails():
    """An invalid gender value should trigger a schema error."""
    df = pd.DataFrame([_make_valid_row(gender="Unknown")])
    with pytest.raises(pa.errors.SchemaError):
        validate_dataframe(df, "test")


def test_negative_tenure_fails():
    """Negative tenure should fail the range check."""
    df = pd.DataFrame([_make_valid_row(tenure_months=-1)])
    with pytest.raises(pa.errors.SchemaError):
        validate_dataframe(df, "test")


def test_invalid_contract_fails():
    """An unknown contract type should fail."""
    df = pd.DataFrame([_make_valid_row(contract="Three year")])
    with pytest.raises(pa.errors.SchemaError):
        validate_dataframe(df, "test")


def test_null_total_charges_passes():
    """total_charges is nullable (new customers may have NaN)."""
    df = pd.DataFrame([_make_valid_row(total_charges=None)])
    result = validate_dataframe(df, "test")
    assert len(result) == 1


def test_churn_binary_only():
    """Churn must be 0 or 1."""
    df = pd.DataFrame([_make_valid_row(churn=2)])
    with pytest.raises(pa.errors.SchemaError):
        validate_dataframe(df, "test")
