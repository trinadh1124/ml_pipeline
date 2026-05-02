"""
test_train.py  —  Unit tests for model training
=================================================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from lightgbm import LGBMClassifier

from tests.conftest import make_sample_df
from src.features.build_features import build_features
from src.models.train import train_model


@pytest.fixture
def train_data():
    df = make_sample_df(100)
    X, _, _ = build_features(df, fit=True)
    y = df["churn"].values
    return X.values, y


def test_model_returns_lgbm_classifier(train_data):
    """train_model should return a fitted LGBMClassifier."""
    X, y = train_data
    model = train_model(X, y)
    assert isinstance(model, LGBMClassifier)


def test_model_predict_shape(train_data):
    """Predictions should match input length."""
    X, y = train_data
    model = train_model(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)


def test_model_predict_proba_range(train_data):
    """Probabilities should be in [0, 1]."""
    X, y = train_data
    model = train_model(X, y)
    probs = model.predict_proba(X)[:, 1]
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_model_custom_params(train_data):
    """Custom parameters should override defaults."""
    X, y = train_data
    model = train_model(X, y, params={"n_estimators": 50, "max_depth": 3})
    assert model.get_params()["n_estimators"] == 50
    assert model.get_params()["max_depth"] == 3


def test_model_custom_scale_pos_weight(train_data):
    """Explicit scale_pos_weight should be used."""
    X, y = train_data
    model = train_model(X, y, scale_pos_weight=5.0)
    assert model.get_params()["scale_pos_weight"] == 5.0
