import numpy as np
from mllib.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
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


def test_confusion_matrix():
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 0, 0], dtype=np.int64)
    y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 0], dtype=np.int64)
    expected_matrix = [[5, 1], [1, 3]]
    result = confusion_matrix(y_true, y_pred)
    assert (result == expected_matrix).all()


def test_confusion_matrix_with_str_labels():
    y_true = np.array(
        ["cat", "dog", "cat", "cat", "dog", "cat", "dog", "dog", "dog", "dog"]
    )
    y_pred = np.array(
        ["cat", "dog", "cat", "dog", "dog", "cat", "dog", "bird", "dog", "dog"]
    )
    labels = ["dog", "cat", "bird"]
    expected_matrix = [[5, 0, 1], [1, 3, 0], [0, 0, 0]]
    result = confusion_matrix(y_true, y_pred, labels=labels)
    assert (result == expected_matrix).all()


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
    with pytest.raises(ValueError):
        confusion_matrix(y_true, y_pred)
