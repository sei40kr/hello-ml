from typing import Any, TypeVar, cast, overload
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


def silhouette_score(
    X: npt.NDArray[np.floating[Any]], labels: npt.NDArray[np.int_ | np.str_]
) -> float:
    """
    Compute the silhouette score.

    The silhouette score for a point i is defined as:
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where:
        a(i) = mean distance to points in same cluster
        b(i) = mean distance to points in nearest different cluster

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data array
    labels : array-like of shape (n_samples,)
        Cluster labels

    Returns
    -------
    silhouette_score : float
        Mean silhouette score across all samples (higher is better)
    """
    n_samples = X.shape[0]

    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        distances[i] = np.sqrt(np.sum((X - X[i]) ** 2, axis=1)) ** 0.5

    silhouette_scores = np.zeros(n_samples)

    for i in range(n_samples):
        same_cluster_mask = (labels == labels[i]) & (np.arange(n_samples) != i)

        a = (
            np.mean(distances[i, same_cluster_mask], dtype=np.float64)
            if np.sum(same_cluster_mask) > 0
            else 0.0
        )
        b = float("inf")
        for label in np.unique(labels):
            if label == labels[i]:
                continue

            other_cluster_mask = labels == label
            if np.sum(other_cluster_mask) > 0:
                mean_dist = np.mean(distances[i, other_cluster_mask])
                b = min(b, mean_dist)

        silhouette_scores[i] = (
            (b - a) / max(a, b) if a != 0 or b != float("inf") else 0.0
        )

    return np.mean(silhouette_scores, dtype=np.float64)
