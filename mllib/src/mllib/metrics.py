from typing import TypeVar, cast, overload
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


@overload
def confusion_matrix(
    y_true: npt.NDArray[np.str_],
    y_pred: npt.NDArray[np.str_],
    labels: list[str] | None = None,
) -> npt.NDArray[np.int64]: ...


@overload
def confusion_matrix(
    y_true: npt.NDArray[np.int64],
    y_pred: npt.NDArray[np.int64],
    labels: None = None,
) -> npt.NDArray[np.int64]: ...


def confusion_matrix(
    y_true: npt.NDArray[np.int64 | np.str_],
    y_pred: npt.NDArray[np.int64 | np.str_],
    labels: list[str] | None = None,
) -> npt.NDArray[np.int64]:
    """
    Compute confusion matrix.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    cm : array-like of shape (n_classes, n_classes)
        Confusion matrix
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape")

    labels_ = (
        cast(list[str], np.unique(np.concatenate([y_true, y_pred])).tolist())
        if labels is None
        else labels
    )

    label_to_index = {label: i for i, label in enumerate(labels_)}
    cm = np.zeros((len(labels_), len(labels_)), dtype=np.int64)

    for true, pred in zip(y_true, y_pred):
        if true in label_to_index and pred in label_to_index:
            cm[label_to_index[true], label_to_index[pred]] += 1

    return cm
