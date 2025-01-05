#!/usr/bin/env nix-shell
#!nix-shell -i python -p "python3.withPackages(ps: with ps; [ numpy matplotlib ])"

from collections import Counter
from typing import Generic, List, TypeVar, Union
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

L = TypeVar("L", bound=Union[np.int64, np.float64, np.str_])


class KNN(Generic[L]):
    """
    K-Nearest Neighbors classifier

    Type Parameters:
        L: The dtype of labels (np.int64, np.float64, or np.str_)
    """

    k: int
    X_train: npt.NDArray[np.float64] | None
    y_train: npt.NDArray[L] | None

    def __init__(self, k: int):
        """
        Initialize the KNN classifier

        Parameters:
            k: Number of neighbors to use for prediction
        """
        self.k = k

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[L]) -> None:
        """
        Store the training data

        Parameters:
            X: Training features
            y: Training labels
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[L]:
        """
        Predict labels for test data

        Parameters:
            X: Test features

        Returns:
            Predicted labels

        Raises:
            RuntimeError: If model hasn't been fitted
        """

        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Model must be fitted before making predictions")

        predictions: List[L] = []

        for x in X:
            distances = [
                self.euclidean_distance(x, x_train) for x_train in self.X_train
            ]
            k_indices = np.argsort(distances)[: self.k]
            k_nearest_labels = self.y_train[k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)

            predictions.append(most_common[0][0])

        return np.array(predictions, dtype=self.y_train.dtype)

    def euclidean_distance(
        self, x1: npt.NDArray[np.float64], x2: npt.NDArray[np.float64]
    ) -> float:
        """
        Calculate the Euclidean distance between two points

        Parameters:
            x1: The first point
            x2: The second point

        Returns:
            Euclidean distance between x1 and x2
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))


def main():
    rng = np.random.default_rng(0)

    X_train = np.concatenate(
        [
            rng.standard_normal(size=(50, 2)) + [2, 2],
            rng.standard_normal(size=(50, 2)) - [2, 2],
        ]
    ).astype(np.float64)
    y_train = np.array([1] * 50 + [2] * 50, dtype=np.int64)

    knn = KNN[np.int64](k=3)
    knn.fit(X_train, y_train)

    x_test = np.array([0, 0], dtype=np.float64)
    y_pred = knn.predict(x_test.reshape(1, -1))  # noqa: F841

    plt.figure(figsize=(10, 8))

    # Plot training points
    scatter = plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap="RdYlBu",
        edgecolors="black",
        s=100,
    )

    # Add test point
    plt.scatter(x_test[0], x_test[1], s=200, c="green", marker="*", label="Test Point")

    # Visualize k nearest neighbors
    distances = np.array(
        [knn.euclidean_distance(x_test, x_train) for x_train in X_train],
        dtype=np.float64,
    )
    k_indices = np.argsort(distances)[: knn.k].astype(np.int64)
    plt.scatter(
        X_train[k_indices, 0],
        X_train[k_indices, 1],
        s=200,
        c="none",
        linewidths=2,
        edgecolors="green",
        label="k Nearest Neighbors",
    )

    # Draw lines to nearest neighbors
    for i in k_indices:
        plt.plot(
            [x_test[0], X_train[i, 0]],
            [x_test[1], X_train[i, 1]],
            "g--",
            alpha=0.3,
        )

    plt.title(f"k-NN Decision Boundary (k={knn.k})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(scatter)

    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
