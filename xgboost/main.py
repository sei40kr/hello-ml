from __future__ import annotations
from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
from mllib.datasets import load_breast_cancer
from mllib.metrics import accuracy_score, f1_score, precision_score, recall_score
from mllib.model_selection import train_test_split


@dataclass
class InternalNode:
    feature_idx: int
    threshold: float
    left: Node
    right: Node


@dataclass
class LeafNode:
    value: float


Node = InternalNode | LeafNode


@dataclass
class XGBoostTree:
    """
    Decision tree for XGBoost.

    The tree follows the following splitting criterion based on the XDGBoost paper:

    Gain = 1/2 [Gₗ²/Hₗ + Gᵣ²/Hᵣ - (Gₗ + Gᵣ)² / (Hₗ + Hᵣ)] + γ

    where:
    - Gₗ, Gᵣ: Sum of gradients of left and right nodes
    - Hₗ, Hᵣ: Sum of hessians of left and right nodes
    - γ: Regularization parameter
    """

    max_depth: int
    min_samples_split: int
    gamma: float
    lambda_: float
    root: Node | None = None

    def fit(
        self,
        X: npt.NDArray[np.float64],
        gradients: npt.NDArray[np.float64],
        hessians: npt.NDArray[np.float64],
    ):
        """
        Fit the tree to the training data using gradients and hessians.

        Args:
            X: Training features matrix of shape (n, m)
            gradients: First-order gradients of shape (n,)
            hesians: Second-order hessians of shape (n,)
        where:
            n: Number of samples
            m: Number of features
        """
        self.root = self._build_tree(X, gradients, hessians, depth=0)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Predicts using the tree.

        Args:
            X: Features matrix of shape (n, m)
        where:
            n: Number of samples
            m: Number of features

        Returns:
            Predictions of shape (n,)
        """
        if self.root is None:
            raise RuntimeError("Tree is not fitted yet")
        return np.array([self._traverse_tree(x, node=self.root) for x in X])

    def _build_tree(
        self,
        X: npt.NDArray[np.float64],
        gradients: npt.NDArray[np.float64],
        hessians: npt.NDArray[np.float64],
        depth: int,
    ) -> Node:
        n, m = X.shape

        # Calculate leaf value using the XGBoost weight formula:
        # w = -G / (H + λ)
        if self.max_depth <= depth or n < self.min_samples_split:
            leaf_value = -np.sum(gradients) / (np.sum(hessians) + self.lambda_)
            return LeafNode(value=leaf_value)

        best_gain = 0.0
        best_split: tuple[int, float, npt.NDArray[np.int64]] | None = None

        for feature_idx in range(m):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                g_l = np.sum(gradients[left_mask])
                g_r = np.sum(gradients[right_mask])
                h_l = np.sum(hessians[left_mask])
                h_r = np.sum(hessians[right_mask])

                gain = (
                    0.5
                    * (
                        (g_l**2) / (h_l + self.lambda_)
                        + (g_r**2) / (h_r + self.lambda_)
                        - ((g_l + g_r) ** 2) / (h_l + h_r + self.lambda_)
                    )
                    + self.gamma
                )

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_idx, threshold, left_mask)

        if best_gain <= 0 or best_split is None:
            return LeafNode(
                value=-np.sum(gradients) / (np.sum(hessians) + self.lambda_)
            )

        feature_idx, threshold, left_mask = best_split

        left = self._build_tree(
            X=X[left_mask],
            gradients=gradients[left_mask],
            hessians=hessians[left_mask],
            depth=depth + 1,
        )
        right = self._build_tree(
            X=X[~left_mask],
            gradients=gradients[~left_mask],
            hessians=hessians[~left_mask],
            depth=depth + 1,
        )

        return InternalNode(feature_idx, threshold, left, right)

    def _traverse_tree(self, x: npt.NDArray[np.float64], node: Node) -> float:
        if isinstance(node, LeafNode):
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


@dataclass
class XGBoost:
    """
    XGBoost classifier implementation.

    The algorithm follows the XGBoost paper's gradient boosting framework:

    obj(θ) = ∑ᵢ L(𝐲ᵢ + 𝐲'ᵢ) + ∑ₖ Ω(𝐟ₖ)

    where:
    - L: Loss function
    - Ω: Regularization term
    - 𝐟ₖ: k-th tree in ensemble

    The model optimizes the objective using Newton-Raphson steps with:
    - First-order gradient: g = ∂𝐲' L(𝐲, 𝐲')
    - Second-order gradient: h = ∂²𝐲' L(𝐲, 𝐲')
    """

    n_estimators: int
    learning_rate: float
    max_depth: int = 3
    min_samples_split: int = 2
    gamma: float = 0.0
    lambda_: float = 1.0

    def __post_init__(self):
        self.trees: list[XGBoostTree] = []
        self.base_score: float = 0.0

    def _logistic_loss(
        self, y: npt.NDArray[np.float64], raw_pred: npt.NDArray[np.float64]
    ) -> float:
        """
        Computes logistic loss for binary classification.
        """
        pred = 1.0 / (1.0 + np.exp(-raw_pred))
        return -np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred), dtype=np.float64)

    def _compute_gradients(
        self, y: npt.NDArray[np.float64], raw_pred: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Computes gradients and hessians for logistic loss.
        """
        pred = 1.0 / (1.0 + np.exp(-raw_pred))
        gradients = (pred - y).astype(np.float64)
        hessians = (pred * (1 - pred)).astype(np.float64)
        return gradients, hessians

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        """
        Fits the XGBoost model to the training data.

        Args:
            X: Training features matrix of shape (n, m)
            y: Training labels of shape (n,)
        where:
            n: Number of samples
            m: Number of features
        """
        self.base_score = np.log(np.mean(y) / (1 - np.mean(y)))
        current_pred = np.full_like(y, self.base_score)

        for _ in range(self.n_estimators):
            gradients, hessians = self._compute_gradients(y, current_pred)

            tree = XGBoostTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                gamma=self.gamma,
                lambda_=self.lambda_,
            )
            tree.fit(X, gradients, hessians)
            self.trees.append(tree)

            current_pred += self.learning_rate * tree.predict(X)

    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Predicts class probabilities.

        Args:
            X: Feature matrix of shape (n, m)

        Returns:
            Class probabilities of shape (n,)
        """
        raw_pred = np.full(X.shape[0], self.base_score, dtype=np.float64)
        for tree in self.trees:
            raw_pred += self.learning_rate * tree.predict(X)
        return (1.0 / (1.0 + np.exp(-raw_pred))).astype(np.float64)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """
        Predicts class labels.

        Args:
            X: Feature matrix of shape (n, m)

        Returns:
            Class labels of shape (n,)
        """
        return (self.predict_proba(X) > 0.5).astype(np.int64)


def main():
    data = load_breast_cancer()
    X, y = data.data.astype(np.float64), data.target.astype(np.float64)

    X_train, X_test, y_train, y_true = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    xgb = XGBoost(n_estimators=100, learning_rate=0.1)
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)

    mse = np.mean((y_true - y_pred) ** 2)
    r2 = 1 - mse / np.var(y_true)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print(f"Precision: {precision_score(y_true, y_pred):.2f}")
    print(f"Recall: {recall_score(y_true, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.2f}")
    # TODO: Show ROC curve and AUC score


if __name__ == "__main__":
    main()
