from __future__ import annotations
from dataclasses import dataclass
import numpy.typing as npt
import numpy as np


@dataclass
class InternalNode:
    """
    Internal node of the decision tree that performs splitting

    Attributes:
        feature_idx: Index of the feature used for splitting
        threshold: Threshold value for splitting
        left: Left child node
        right: Right child node
        depth: Depth of the node in the tree
    """

    feature_idx: int
    threshold: float
    left: Node
    right: Node
    depth: int


@dataclass
class LeafNode:
    """
    Leaf node of the decision tree that holds a prediction value

    Attributes:
        value: Predicted value
        depth: Depth of the node in the tree
    """

    value: float
    depth: int


Node = InternalNode | LeafNode


class DecisionTree:
    root: Node | None
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int

    def __init__(
        self, max_depth: int = 3, min_samples_split: int = 2, min_samples_leaf: int = 1
    ):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        """
        Fit the decision tree to the training data

        Args:
            X: Feature matrix of shape (n, m)
            y: Target array of shape (n)
        where:
            n: Number of samples
            m: Number of features
        """
        self.n, self.m = X.shape
        self.root = self._grow_tree(X, y)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Make predictions

        Args:
            X: Feature matrix of shape (n, m)
        where:
            n: Number of samples
            m: Number of features

        Returns:
            npt.NDArray[np.float64]: Arary of predictions of shape (n)
        """
        if self.root is None:
            raise RuntimeError("Tree not fitted yet")
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(
        self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64], depth: int = 0
    ) -> Node:
        """
        Recursively grow the decision tree

        Args:
            X: Feature matrix of shape (n, m)
            y: Target array of shape (n)
            depth: Current depth of the node

        Returns:
            Node: The created Node
        """
        n, _ = X.shape

        if (
            self.max_depth <= depth
            or n < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return self._create_leaf(y, depth)

        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None or threshold is None:
            return self._create_leaf(y, depth)

        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        if (np.sum(left_mask) < self.min_samples_leaf) or (
            np.sum(right_mask) < self.min_samples_leaf
        ):
            return self._create_leaf(y, depth)

        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return InternalNode(
            feature_idx=feature_idx,
            threshold=threshold,
            left=left,
            right=right,
            depth=depth,
        )

    def _create_leaf(self, y: npt.NDArray[np.float64], depth: int) -> LeafNode:
        return LeafNode(value=np.mean(y, dtype=np.float64), depth=depth)

    def _best_split(
        self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> tuple[int | None, float | None]:
        """
        Find the best split point

        Args:
            X: Feature matrix of shape (n, m)
            y: Target array of shape (n)
        where:
            n: Number of samples
            m: Number of features

        Returns:
            tuple[int | None, float | None]:
                Best feature index and threshold.
                Returns None, None if no valid point is found.
        """
        best_gain = -1
        best_feature_idx = None
        best_threshold = None

        current_impurity = self._calculate_variance(y)

        for feature_idx in range(self.m):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if (
                    np.sum(left_mask) < self.min_samples_leaf
                    or np.sum(right_mask) < self.min_samples_leaf
                ):
                    continue

                gain = self._calculate_gain(
                    y, y[left_mask], y[right_mask], current_impurity
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def _calculate_variance(self, y: npt.NDArray[np.float64]) -> float:
        """
        Calculate mean squared error

        Args:
            y: Target array of shape (n)
        where:
            n: Number of samples

        Returns:
            float: Mean squared error
        """
        return np.mean((y - np.mean(y, dtype=np.float64)) ** 2, dtype=np.float64)

    def _calculate_gain(
        self,
        parent: npt.NDArray[np.float64],
        left: npt.NDArray[np.float64],
        right: npt.NDArray[np.float64],
        current_impurity: float,
    ) -> float:
        """
        Calculate information gain

        Args:
            parent: Parent node target array of shape (n)
            left: Left child node target array of shape (n)
            right: Right child node target array of shape (n)
            current_impurity: Current node impurity
        """
        n = len(parent)
        n_l, n_r = len(left), len(right)

        if n_l == 0 or n_r == 0:
            return 0

        impurity_left = self._calculate_variance(left)
        impurity_right = self._calculate_variance(right)
        return current_impurity - (n_l / n) * impurity_left - (n_r / n) * impurity_right

    def _traverse_tree(self, x: npt.NDArray[np.float64], node: Node) -> float:
        """
        Traverse the tree to make a prediction

        Args:
            x: Single sample feature vector of shape (m)
            node: Current node in the tree

        Returns:
            float: Predicted value
        """
        if isinstance(node, LeafNode):
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node=node.left)
        return self._traverse_tree(x, node=node.right)
