from typing import TypeVar
import numpy as np
import numpy.typing as npt

T = TypeVar("T", bound=np.generic)


def accuracy_score(y_true: npt.NDArray[T], y_pred: npt.NDArray[T]) -> float:
    """
    Compute the accuracy.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels

    Returns
    -------
    score : float
        The fraction of correctly classified samples
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape")
    return np.mean(y_true == y_pred)


def precision_score(y_true: npt.NDArray[T], y_pred: npt.NDArray[T]) -> float:
    """
    Compute the precision.

    Precision = TP / (TP + FP)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels

    Returns
    -------
    score : float
        Precision score
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape")
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if 0 < predicted_positives else 0.0


def recall_score(y_true: npt.NDArray[T], y_pred: npt.NDArray[T]) -> float:
    """
    Compute the recall.

    Recall = TP / (TP + FN)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels

    Returns
    -------
    score : float
        Recall score
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape")
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if 0 < actual_positives else 0.0


def f1_score(y_true: npt.NDArray[T], y_pred: npt.NDArray[T]) -> float:
    """
    Compute F1 score.

    F1 = 2 / (1 / Precision + 1 / Recall)
       = 2 * Precision * Recall / (Precision + Recall)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels

    Returns
    -------
    score : float
        F1 score
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return (
        2 * precision * recall / (precision + recall) if 0 < precision + recall else 0.0
    )
