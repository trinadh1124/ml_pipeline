import pandas as pd

from src.data.loader import _create_monitoring_splits, resolve_batch_path


def _make_split_source() -> pd.DataFrame:
    rows = []
    for i in range(650):
        rows.append({
            "customer_id": f"S4-{i:04d}",
            "gender": "Female",
            "senior_citizen": "No",
            "partner": "No",
            "dependents": "No",
            "tenure_months": 6,
            "phone_service": "Yes",
            "multiple_lines": "Yes",
            "internet_service": "Fiber optic",
            "online_security": "No",
            "online_backup": "No",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "Yes",
            "streaming_movies": "Yes",
            "contract": "Month-to-month",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 95.0,
            "total_charges": 500.0,
            "churn": 1,
        })

    for i in range(350):
        rows.append({
            "customer_id": f"S3-{i:04d}",
            "gender": "Male",
            "senior_citizen": "No",
            "partner": "Yes",
            "dependents": "No",
            "tenure_months": 10,
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
            "payment_method": "Mailed check",
            "monthly_charges": 80.0,
            "total_charges": 800.0,
            "churn": 1 if i < 200 else 0,
        })

    for i in range(6100):
        rows.append({
            "customer_id": f"B-{i:04d}",
            "gender": "Male" if i % 2 else "Female",
            "senior_citizen": "No",
            "partner": "Yes" if i % 3 else "No",
            "dependents": "No",
            "tenure_months": 24 + (i % 36),
            "phone_service": "Yes",
            "multiple_lines": "No",
            "internet_service": "DSL" if i % 2 else "No",
            "online_security": "Yes" if i % 2 else "No internet service",
            "online_backup": "Yes" if i % 2 else "No internet service",
            "device_protection": "Yes" if i % 2 else "No internet service",
            "tech_support": "Yes" if i % 2 else "No internet service",
            "streaming_tv": "No" if i % 2 else "No internet service",
            "streaming_movies": "No" if i % 2 else "No internet service",
            "contract": "One year" if i % 2 else "Two year",
            "paperless_billing": "No",
            "payment_method": "Bank transfer (automatic)" if i % 2 else "Credit card (automatic)",
            "monthly_charges": 45.0 + (i % 15),
            "total_charges": 1400.0 + (i % 500),
            "churn": 1 if i % 8 == 0 else 0,
        })
    return pd.DataFrame(rows)


def test_create_monitoring_splits_creates_expected_sizes():
    df = _make_split_source()

    train, current, stress = _create_monitoring_splits(df)

    assert len(df) == 7100
    assert len(stress) == 800
    assert len(current) == 943
    assert len(train) == len(df) - len(stress) - len(current)
    assert stress["churn"].mean() > current["churn"].mean()
    assert stress["churn"].mean() > train["churn"].mean()


def test_resolve_batch_path_supports_stress():
    assert resolve_batch_path("current").name == "current.csv"
    assert resolve_batch_path("stress").name == "current_stress.csv"
