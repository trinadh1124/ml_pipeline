"""
test_evaluate.py  —  Unit tests for evaluation and deployment gates
====================================================================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.models.evaluate import compute_metrics, check_gates


def test_perfect_predictions():
    """Perfect predictions should give F1 = 1.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    m = compute_metrics(y_true, y_pred, y_prob)
    assert m["f1"] == 1.0
    assert m["recall"] == 1.0
    assert m["missed_churners"] == 0
    assert m["total_business_cost"] == 0


def test_all_wrong_predictions():
    """All-wrong predictions should give recall = 0."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 0])
    y_prob = np.array([0.8, 0.8, 0.2, 0.2, 0.2])
    m = compute_metrics(y_true, y_pred, y_prob)
    assert m["recall"] == 0.0
    assert m["missed_churners"] == 3
    assert m["business_cost_fn"] == 3 * 1500


def test_business_cost_calculation():
    """Verify cost model: FN * $1500 + FP * $50."""
    y_true = np.array([1, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 0])  # 2 FN, 1 FP
    y_prob = np.array([0.9, 0.3, 0.3, 0.7, 0.2])
    m = compute_metrics(y_true, y_pred, y_prob)
    assert m["fn"] == 2
    assert m["fp"] == 1
    assert m["total_business_cost"] == 2 * 1500 + 1 * 50


def test_gates_pass_with_strong_metrics():
    """Good metrics should pass all gates."""
    metrics = {"f1": 0.70, "recall": 0.80, "auc_pr": 0.60, "auc_roc": 0.85}
    passed, failures = check_gates(metrics)
    assert passed is True
    assert failures == []


def test_gates_fail_with_low_f1():
    """F1 below threshold should fail."""
    metrics = {"f1": 0.30, "recall": 0.80, "auc_pr": 0.60, "auc_roc": 0.85}
    passed, failures = check_gates(metrics)
    assert passed is False
    assert any("F1" in f for f in failures)


def test_gates_fail_with_low_recall():
    """Recall below threshold should fail."""
    metrics = {"f1": 0.70, "recall": 0.40, "auc_pr": 0.60, "auc_roc": 0.85}
    passed, failures = check_gates(metrics)
    assert passed is False
    assert any("Recall" in f for f in failures)


def test_gates_multiple_failures():
    """Multiple failing metrics should all be reported."""
    metrics = {"f1": 0.10, "recall": 0.10, "auc_pr": 0.10, "auc_roc": 0.50}
    passed, failures = check_gates(metrics)
    assert passed is False
    assert len(failures) == 4
