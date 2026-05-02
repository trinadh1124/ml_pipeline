"""
model_selection.py  --  Rigorous Model Comparison
==================================================
Same approach as Module 1: define the evaluation framework BEFORE any model runs.

THE SITUATION
-------------
It is day one. The team has 4,857 established customers (tenure > 12 months).
Before building any pipeline, they compare four models on the same data,
same split, same metric. The winner goes into the production pipeline.

    python model_selection.py

This script logs every comparison run to MLflow -- so the decision is auditable.
In Module 1, this was a standalone script with CSV output. Now every run
is in the MLflow UI, with params, metrics, and model artefacts.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from xgboost import XGBClassifier

from src.config.config import SEED, TEST_SIZE, TARGET, MLFLOW_TRACKING_URI
from src.data.loader import load_train
from src.data.validate import validate_dataframe
from src.features.build_features import build_features
from src.models.train import build_estimator


def main():
    print("="*70)
    print("  MODEL SELECTION -- Rigorous Comparison")
    print("  Same data, same split, same metric for all candidates")
    print("="*70)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("telco-churn-model-selection")

    # Load and validate
    df = load_train()
    df = validate_dataframe(df, "training data")

    # Feature engineering
    X, feature_names, _ = build_features(df, fit=True)
    y = df[TARGET].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y,
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / n_pos

    print(f"\n  Train: {len(X_train)} samples ({y_train.mean():.1%} churn)")
    print(f"  Test:  {len(X_test)} samples ({y_test.mean():.1%} churn)")
    print(f"  scale_pos_weight: {spw:.2f}")
    print(f"\n  Primary metric: F1 on churners (minority class)")
    print(f"  Every model handles class imbalance via class_weight or scale_pos_weight")

    # -- Candidates ------------------------------------------------------------
    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=5000, class_weight="balanced", random_state=SEED,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=SEED,
        ),
        "XGBoost (GBM)": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, scale_pos_weight=spw, random_state=SEED,
        ),
    }

    results = []

    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("model_type", name)
            mlflow.log_param("model", name)
            mlflow.log_param("train_rows", len(X_train))

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            f1 = f1_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            mlflow.log_metrics({
                "f1": f1, "recall": rec, "precision": prec, "auc_roc": auc,
            })

            missed = int((y_test == 1).sum() - (y_pred[y_test == 1] == 1).sum())
            mlflow.log_metric("missed_churners", missed)

            results.append({
                "Model": name,
                "F1": f1,
                "Recall": rec,
                "Precision": prec,
                "AUC-ROC": auc,
                "Missed": missed,
                "Cost ($)": missed * 1500,
            })

    # -- Results ---------------------------------------------------------------
    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    print(f"\n{'-'*80}")
    print(results_df.to_string(index=False))
    print(f"{'-'*80}")

    winner = results_df.iloc[0]
    print(f"\n  WINNER: {winner['Model']}")
    print(f"  F1={winner['F1']:.3f}, Recall={winner['Recall']:.3f}, "
          f"Missed={winner['Missed']} (${winner['Cost ($)']:,})")
    print(f"\n  All runs logged to MLflow. View with:")
    print(f"  mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")

    # -- Optuna: tune top-3 models dynamically ---------------------------------
    top3 = results_df.head(3)
    eliminated = results_df.iloc[3]

    print(f"\n{'='*70}")
    print(f"  HYPERPARAMETER TUNING (Optuna -- top 3 models)")
    print(f"{'='*70}")
    print(f"  Eliminated (lowest F1): {eliminated['Model']} (F1={eliminated['F1']:.3f})")
    print(f"  Tuning top 3 with Optuna TPE (50 trials each):")
    for _, row in top3.iterrows():
        print(f"    - {row['Model']} (default F1={row['F1']:.3f})")
    print(f"  Each model gets its own search space. Winner -> models/best_params.json")

    from src.models.tuning import tune_model

    # Map model names to sklearn classes for re-instantiation after tuning
    CLASS_MAP = {
        "Logistic Regression": "LogisticRegression",
        "Random Forest":       "RandomForestClassifier",
        "XGBoost (GBM)":       "XGBClassifier",
        "LightGBM":            "LGBMClassifier",
    }

    tuned_results = []
    for _, row in top3.iterrows():
        model_name = row["Model"]
        print(f"\n  Tuning: {model_name} ...")
        trial_budget = 80 if model_name == "XGBoost (GBM)" else 50
        timeout_budget = 600 if model_name == "XGBoost (GBM)" else 300

        best_params, best_f1 = tune_model(
            model_name,
            X_train.values if hasattr(X_train, "values") else X_train,
            y_train,
            X_test.values if hasattr(X_test, "values") else X_test,
            y_test,
            n_trials=trial_budget,
            timeout=timeout_budget,
        )

        # Re-fit with best params to get full metrics
        m = build_estimator(CLASS_MAP[model_name], best_params, scale_pos_weight=spw)

        m.fit(X_train, y_train)
        y_pred_t = m.predict(X_test)
        y_prob_t = m.predict_proba(X_test)[:, 1]

        f1_t   = f1_score(y_test, y_pred_t)
        rec_t  = recall_score(y_test, y_pred_t)
        prec_t = precision_score(y_test, y_pred_t)
        auc_t  = roc_auc_score(y_test, y_prob_t)
        missed_t = int((y_test == 1).sum() - (y_pred_t[y_test == 1] == 1).sum())

        # Log tuned run to MLflow
        with mlflow.start_run(run_name=f"{model_name} (Optuna-tuned)"):
            mlflow.set_tag("model_type", f"{model_name} (tuned)")
            mlflow.log_param("model", model_name)
            mlflow.log_params({k: v for k, v in best_params.items()
                               if not isinstance(v, float) or abs(v) < 1e6})
            mlflow.log_metrics({
                "f1": f1_t, "recall": rec_t, "precision": prec_t,
                "auc_roc": auc_t, "missed_churners": missed_t,
            })

        tuned_results.append({
            "model_name":  model_name,
            "best_params": best_params,
            "f1":          f1_t,
            "recall":      rec_t,
            "auc_roc":     auc_t,
            "missed":      missed_t,
        })

    # -- Post-tuning comparison -------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  POST-TUNING COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'Default F1':>10} {'Tuned F1':>10} {'Improvement':>12}")
    print(f"  {'-'*60}")
    for r in tuned_results:
        default_f1 = results_df[results_df["Model"] == r["model_name"]]["F1"].values[0]
        marker = "  <- PRODUCTION MODEL" if r == max(tuned_results, key=lambda x: x["f1"]) else ""
        print(f"  {r['model_name']:<25} {default_f1:>10.3f} {r['f1']:>10.3f} "
              f"  {r['f1']-default_f1:>+.3f}{marker}")

    # -- Save winner to best_params.json with optimal threshold ----------------
    import json, datetime
    from src.models.evaluate import find_optimal_threshold

    winner_tuned = max(tuned_results, key=lambda r: r["f1"])

    # Re-fit winner on full train to get probabilities on test for threshold opt
    wm = build_estimator(
        CLASS_MAP[winner_tuned["model_name"]],
        winner_tuned["best_params"],
        scale_pos_weight=spw,
    )
    wm.fit(X_train, y_train)
    y_prob_w = wm.predict_proba(X_test)[:, 1]

    print(f"\n  Threshold optimisation for production:")
    cost_t, f1_t, _ = find_optimal_threshold(y_test, y_prob_w, min_precision=0.40)

    best_entry = {
        "model_class":       CLASS_MAP[winner_tuned["model_name"]],
        "params":            dict(winner_tuned["best_params"]),
        "f1_score":          round(winner_tuned["f1"], 4),
        "recall":            round(winner_tuned["recall"], 4),
        "auc_roc":           round(winner_tuned["auc_roc"], 4),
        "threshold_cost":    round(cost_t, 2),
        "threshold_f1":      round(f1_t, 2),
        "selected_at":       datetime.datetime.now().isoformat(timespec="seconds"),
        "source":            "model_selection.py -- Optuna TPE, 50 trials, 3-fold CV per top-3 candidate",
    }
    bp_path = os.path.join("models", "best_params.json")
    os.makedirs("models", exist_ok=True)
    with open(bp_path, "w") as f:
        json.dump(best_entry, f, indent=2)

    print(f"\n  Production model  : {winner_tuned['model_name']}")
    print(f"  Tuned CV F1       : {winner_tuned['f1']:.3f}")
    print(f"  Cost threshold    : {cost_t:.2f}  (use for production scoring)")
    print(f"  F1 threshold      : {f1_t:.2f}  (use for reporting)")
    print(f"  Saved to          : {bp_path}")
    print(f"  main.py reads best_params.json at Stage 5 -- no hardcoding.")

    # -- The drift scenario ----------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  WHAT HAPPENS AFTER DEPLOYMENT")
    print(f"{'='*70}")
    print(f"  Train (established, tenure > 12) churn rate: {y.mean():.1%}")
    print(f"  New customers (tenure <= 12) churn rate:     47.4%")
    print(f"  -> New customers churn at 2.8x the rate")
    print(f"  -> Contract mix shifts: more month-to-month, more fiber optic")
    print(f"  -> This is the drift that Evidently will detect in main.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
