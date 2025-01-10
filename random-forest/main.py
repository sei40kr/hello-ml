#!/usr/bin/env python3

import numpy.typing as npt
import numpy as np
import pandas as pd
import decisiontree as dt


class RandomForest:
    """
    Random Forest class for regression tasks

    Attributes:
        n_trees: Number of trees
        min_samples_split: Minimum number of samples required for splitting
        max_depth: Maximum depth of trees
    """

    n_trees: int
    min_samples_split: int
    max_depth: int

    def __init__(
        self, n_trees: int = 10, min_samples_split: int = 2, max_depth: int = 5
    ):
        """
        Args:
            n_trees: Number of trees
            min_samples_split: Minimum number of samples required for splitting
            max_depth: Maximum depth of trees
        """
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []

    def fit(
        self,
        rng: np.random.Generator,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> None:
        """
        Train the random forest

        Args:
            X: Feature matrix of shape (n, m)
            y: Target array of shape (n,)
        where:
            n: Number of samples
            m: Number of features
        """
        self.trees = []

        for _ in range(self.n_trees):
            tree = dt.DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            indices = rng.choice(X.shape[0], size=X.shape[0], replace=True)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Make predictions

        Args:
            X: Feature matrix of shape (n, m)
        where:
            n: Number of samples
            m: Number of features

        Returns:
            npt.NDArray[np.float64]: Array of predictions of shape (n,)
        """
        return np.mean([tree.predict(X) for tree in self.trees], axis=0)


def main():
    rng = np.random.default_rng(0)

    df = pd.read_csv("../datasets/boston-housing/boston-housing.csv")
    df = df.sample(frac=1.0, random_state=0)

    X = df.drop("medv", axis=1).values.astype(np.float64)
    y = df["medv"].values.astype(np.float64)

    test_size = 0.3
    test_count = int(len(df) * test_size)

    X_train, X_test = X[:-test_count], X[-test_count:]
    y_train, y_true = y[:-test_count], y[-test_count:]

    rf = RandomForest(n_trees=100, max_depth=3, min_samples_split=5)
    rf.fit(rng, X_train, y_train)

    y_pred = rf.predict(X_test)

    mse = np.mean((y_true - y_pred) ** 2)
    r2 = 1 - mse / np.var(y_true)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")


if __name__ == "__main__":
    main()
