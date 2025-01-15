import numpy as np
from mllib.metrics import accuracy_score, f1_score, precision_score, recall_score
import pytest


def test_accuracy_score():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    assert accuracy_score(y_true, y_pred) == 0.8


def test_precision_score():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    assert precision_score(y_true, y_pred) == 1.0


def test_recall_score():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    assert recall_score(y_true, y_pred) == 0.6666666666666666


def test_f1_score():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    assert f1_score(y_true, y_pred) == 0.8


def test_shape_mismatch():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0])
    with pytest.raises(ValueError):
        accuracy_score(y_true, y_pred)
    with pytest.raises(ValueError):
        precision_score(y_true, y_pred)
    with pytest.raises(ValueError):
        recall_score(y_true, y_pred)
    with pytest.raises(ValueError):
        f1_score(y_true, y_pred)
