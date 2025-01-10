#!/usr/bin/env nix-shell
#!nix-shell -i python -p "python3.withPackages(ps: with ps; [ matplotlib numpy pandas ])"


from typing import cast
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

from decisiontree import DecisionTree, InternalNode, LeafNode, Node


def main():
    df = read_csv("../datasets/boston-housing/boston-housing.csv")
    df = df.sample(frac=1.0, random_state=0)

    X = df.drop("medv", axis=1).values.astype(np.float64)
    y = df["medv"].values.astype(np.float64)

    test_size = 0.3
    test_count = int(len(df) * test_size)

    X_train, X_test = X[:-test_count], X[-test_count:]
    y_train, y_true = y[:-test_count], y[-test_count:]

    tree = DecisionTree(max_depth=3, min_samples_split=5, min_samples_leaf=2)
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)

    mse = np.mean((y_true - y_pred) ** 2)
    r2 = 1 - mse / np.var(y_true)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    feature_names = df.columns

    def plot_node(node: Node, x: float, y: float, dx: float, dy: float, ax: Axes):
        """
        Plot a node in the decision tree

        Args:
            node: Node to plot
            x: The x-coordinate of the node
            y: The y-coordinate of the node
            dx: The horizontal distance to the child nodes
            dy: The vertical distance to the child nodes
            ax: The matplotlib Axes object to plot on
        """
        if isinstance(node, LeafNode):
            text = f"value = {node.value:.2f}"
            bbox = {"facecolor": "lightgreen", "edgecolor": "black"}
        else:
            text = f"{feature_names[node.feature_idx]}\n≦ {node.threshold:.2f}"
            bbox = {"facecolor": "lightblue", "edgecolor": "black"}

        ax.text(x, y, text, bbox=bbox, ha="center", va="center")

        if isinstance(node, InternalNode):
            ax.arrow(x, y - 0.02, -dx, -dy, head_width=0.02)
            plot_node(node.left, x - dx, y - dy - 0.1, dx / 2, dy, ax)
            ax.arrow(x, y - 0.02, dx, -dy, head_width=0.02)
            plot_node(node.right, x + dx, y - dy - 0.1, dx / 2, dy, ax)

    _, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    plot_node(cast(Node, tree.root), 0, 0.9, 0.4, 0.2, ax)
    plt.title("Decision Tree Visualization")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
