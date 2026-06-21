"""
baseline.py  --  First Pass (The Naive Approach)
=================================================
A junior data scientist builds a quick churn predictor.
The code runs. The numbers look fine. The problems are invisible.

THE SITUATION
-------------
A telecom company is losing 26% of its customers. Leadership wants a
churn prediction model deployed immediately. The DS team has data on
7,043 customers -- demographics, service subscriptions, billing, and
whether they churned.

A junior analyst grabs the dataset and builds a model in 30 minutes.
Here is what goes wrong.

    python baseline.py

PITFALLS DEMONSTRATED
---------------------
  1. Hardcoded paths
  2. LabelEncoder on nominal categories (contract type is NOT ordinal)
  3. No temporal split -- model sees new-customer patterns during training
  4. Optimised for accuracy (73% churn rate in majority class -> misleading)
  5. No feature pipeline saved -- inference will fail on unseen categories
  6. No experiment tracking -- no record of what was tried
  7. No drift monitoring -- model silently degrades on new customer segments
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

# ── PITFALL 1: Hardcoded path ────────────────────────────────────────────────
# On the analyst's machine this works. On CI, Docker, a colleague's laptop: crash.
DATA_PATH = "C:/Users/trina/Downloads/MLOps_Pipeline_with_Tools/M2_MLOps_Pipeline/data/telco_data.csv"

# Fallback for demo purposes
import os
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "telco_churn_full.csv")

print(DATA_PATH)

print("="*70)
print("  BASELINE -- The Naive Approach")
print("  What a quick-and-dirty churn model looks like")
print("="*70)

df = pd.read_csv(DATA_PATH)
df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
df = df.dropna()
print(f"\n  Loaded {len(df)} customers")
print(f"  Churn rate: {df['churn'].mean():.1%}")

# ── PITFALL 2: LabelEncoder on nominal categories ────────────────────────────
# Contract types are NOMINAL (Month-to-month, One year, Two year).
# LabelEncoder assigns 0, 1, 2 -- implying Month-to-month < One year < Two year.
# This is technically true for commitment level, but LabelEncoder doesn't know that.
# For 'payment_method' with 4 values, the ordering is completely arbitrary.
print("\n  Encoding categoricals with LabelEncoder...")
le_cols = []
for col in df.select_dtypes(include="object").columns:
    if col == "customer_id":
        continue
    le = LabelEncoder() 
    df[col] = le.fit_transform(df[col].astype(str))
    le_cols.append(col)
print(f"  LabelEncoded {len(le_cols)} columns: {le_cols[:5]}...")

# ── PITFALL 3: No temporal split ─────────────────────────────────────────────
# New customers (tenure <= 12) have VERY different churn patterns (47% vs 17%).
# Mixing them in training lets the model memorise new-customer patterns.
# In production, when the new-customer cohort shifts, the model breaks silently.
features = [c for c in df.columns if c not in ["customer_id", "churn"]]
X = df[features]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
)
print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")
print(f"  (No temporal awareness -- new and old customers mixed freely)")

# ── PITFALL 4: Optimise for accuracy ─────────────────────────────────────────
print("\n  Training GradientBoosting (default params, no tuning)...")
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print(f"\n{'-'*50}")
print(f"  RESULTS -- What gets reported to stakeholders:")
print(f"{'-'*50}")
print(f"  Accuracy : {acc:.1%}  <- 'the model is {acc:.0%} accurate!'")
print(f"{'-'*50}")

# ── What DOESN'T get reported ────────────────────────────────────────────────
missed = int((y_test == 1).sum() - (y_pred[y_test == 1] == 1).sum())
cost_missed = missed * 1500
print(f"\n  What doesn't get mentioned:")
print(f"  F1 on churners    : {f1:.3f}")
print(f"  Recall on churners: {rec:.3f}")
print(f"  Missed churners   : {missed} out of {int((y_test==1).sum())}")
print(f"  Revenue at risk   : ${cost_missed:,} in acquisition costs")

# ── PITFALL 5: No feature pipeline saved ──────────────────────────────────────
print(f"\n  Feature pipeline: NOT SAVED")
print(f"  -> LabelEncoders are in memory only. If a new customer segment")
print(f"    appears at inference time (e.g., new payment method), the model crashes.")

# ── PITFALL 6: No experiment tracking ─────────────────────────────────────────
print(f"\n  Experiment log: NONE")
print(f"  -> No record of parameters, metrics, or data version.")
print(f"    Cannot compare this run with future runs.")

# ── PITFALL 7: No drift monitoring ────────────────────────────────────────────
print(f"\n  Drift monitoring: NONE")
print(f"  -> When new-customer demographics shift (more fiber optic,")
print(f"    different contract mix), the model degrades silently.")

print(f"\n{'='*70}")
print(f"  NEXT: model_selection.py -- do this properly")
print(f"{'='*70}")

#added a comment to test the flow
