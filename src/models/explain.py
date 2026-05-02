"""
explain.py  -  Model Interpretability with SHAP
================================================
Answers the business question: "Why was this customer flagged as a churner?"

LightGBM/XGBoost feature importance shows which features the model uses most.
SHAP shows how each feature pushes a specific prediction toward or away
from churn - per-customer explanations, not just global rankings.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap


def explain_model(
    model,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str],
    save_dir: str | Path = "logs",
    max_display: int = 15,
) -> dict:
    """
    Generate SHAP explanations for a trained model.

    Creates:
      - Global feature importance (mean |SHAP|)
      - SHAP summary plot saved as PNG
      - Top feature contributions for interpretation
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    shap_values = explainer.shap_values(X_arr)
    shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    importance = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda item: item[1],
        reverse=True,
    )

    print(f"\n{'-' * 50}")
    print(f"  SHAP Feature Importance (top {min(max_display, len(importance))})")
    print(f"{'-' * 50}")
    for name, value in importance[:max_display]:
        bar = "#" * int(value / importance[0][1] * 30)
        print(f"  {name:30s} {value:.4f}  {bar}")
    print(f"{'-' * 50}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_vals,
        X_arr,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=(10, 6),
    )
    plt.tight_layout()
    plot_path = save_dir / "shap_summary.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [shap] Summary plot saved -> {plot_path}")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_vals,
        X_arr,
        feature_names=feature_names,
        max_display=max_display,
        plot_type="bar",
        show=False,
        plot_size=(10, 6),
    )
    plt.tight_layout()
    bar_path = save_dir / "shap_importance.png"
    plt.savefig(str(bar_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [shap] Importance bar plot saved -> {bar_path}")

    return {
        "top_features": [(name, float(value)) for name, value in importance[:max_display]],
        "summary_plot": str(plot_path),
        "bar_plot": str(bar_path),
        "n_samples_explained": len(X_arr),
    }


def explain_single_customer(
    model,
    customer_row: np.ndarray | pd.DataFrame,
    feature_names: list[str],
) -> dict:
    """
    Explain a single customer's churn prediction.

    Returns the top features pushing toward or away from churn.
    """
    explainer = shap.TreeExplainer(model)
    X = customer_row.values if isinstance(customer_row, pd.DataFrame) else customer_row

    if X.ndim == 1:
        X = X.reshape(1, -1)

    shap_values = explainer.shap_values(X)
    shap_row = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0, 1])

    contributions = sorted(
        zip(feature_names, shap_row, X[0]),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    return {
        "prediction": prediction,
        "probability": probability,
        "top_contributors": [
            {
                "feature": name,
                "shap_value": float(shap_value),
                "feature_value": float(feature_value),
                "direction": "toward churn" if shap_value > 0 else "away from churn",
            }
            for name, shap_value, feature_value in contributions[:10]
        ],
    }
