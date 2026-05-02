"""
validate.py  —  Data Validation with Pandera
==============================================
MODULE 2 KEY TOOL: Pandera replaces the manual schema checks from Module 1.

In Module 1, we wrote:
    if df.isnull().sum().sum() > threshold: abort()
    if not set(df.columns) >= expected: abort()

Now Pandera does this declaratively — you define the schema once,
and it validates automatically on every pipeline run.
"""

import pandas as pd
import pandera.pandas as pa
from pandera import Check
from pandera.pandas import Column, DataFrameSchema


# ── Schema Definition ─────────────────────────────────────────────────────────
# This is the "read-only example schema" the design doc asks us to scaffold.
# Learners tweak 1-2 constraints to match their dataset.

telco_schema = DataFrameSchema(
    columns={
        "customer_id":       Column(str,     nullable=False),
        "gender":            Column(str,     Check.isin(["Male", "Female"])),
        "senior_citizen":    Column(str,     Check.isin(["Yes", "No"])),
        "partner":           Column(str,     Check.isin(["Yes", "No"])),
        "dependents":        Column(str,     Check.isin(["Yes", "No"])),
        "tenure_months":     Column(int,     Check.in_range(0, 72)),
        "phone_service":     Column(str,     Check.isin(["Yes", "No"])),
        "multiple_lines":    Column(str,     Check.isin(["Yes", "No", "No phone service"])),
        "internet_service":  Column(str,     Check.isin(["DSL", "Fiber optic", "No"])),
        "online_security":   Column(str,     Check.isin(["Yes", "No", "No internet service"])),
        "online_backup":     Column(str,     Check.isin(["Yes", "No", "No internet service"])),
        "device_protection": Column(str,     Check.isin(["Yes", "No", "No internet service"])),
        "tech_support":      Column(str,     Check.isin(["Yes", "No", "No internet service"])),
        "streaming_tv":      Column(str,     Check.isin(["Yes", "No", "No internet service"])),
        "streaming_movies":  Column(str,     Check.isin(["Yes", "No", "No internet service"])),
        "contract":          Column(str,     Check.isin(["Month-to-month", "One year", "Two year"])),
        "paperless_billing": Column(str,     Check.isin(["Yes", "No"])),
        "payment_method":    Column(str,     Check.isin([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])),
        "monthly_charges":   Column(float,   Check.in_range(0, 200)),
        "total_charges":     Column(float,   Check.in_range(0, 10_000), nullable=True),
        "churn":             Column(int,     Check.isin([0, 1])),
    },
    coerce=True,
    strict=False,   # allow extra columns without failing
)


def validate_dataframe(df: pd.DataFrame, name: str = "data") -> pd.DataFrame:
    """
    Validate a DataFrame against the Pandera schema.
    Returns the validated (and coerced) DataFrame.
    Raises pa.errors.SchemaError on failure.
    """
    print(f"  [pandera] Validating {name} ({len(df)} rows, {len(df.columns)} cols)...", end=" ")
    try:
        validated = telco_schema.validate(df)
        print("PASSED")
        return validated
    except pa.errors.SchemaError as e:
        print("FAILED")
        print(f"  [pandera] Schema violation:\n{e}")
        raise


def validate_inference_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate inference-time input (no churn column expected)."""
    inference_schema = telco_schema.remove_columns(["churn"])
    return inference_schema.validate(df, lazy=True)
