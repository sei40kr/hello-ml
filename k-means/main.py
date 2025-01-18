from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from mllib.datasets import load_iris
import matplotlib.pyplot as plt


@dataclass
class KMeanResult:
    """
    Results of K-means clustering algorithm.
    """

    labels: npt.NDArray[np.int64]
    centroids: npt.NDArray[np.float64]
    n_iter: int
    inertia: float


class KMeans:
    """
    K-means clustering algorithm.
    """

    n_clusters: int
    max_iter: int
    tol: float
    random_state: int | None

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ):
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive integer")
        if max_iter < 1:
            raise ValueError("max_iter must be positive integer")
        if tol < 0:
            raise ValueError("tol must be non-negative float")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X: npt.NDArray[np.float64]) -> KMeanResult:
        """
        Fit K-means clustering.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            KMeanResult containing:
            - labels: Cluster labels for each point
            - centroids: Final centroids positions
            - n_iter: Number of iterations run
            - inertia: Final inertia value
        """
        labels = np.zeros(X.shape[0], dtype=np.int64)
        centroids = self._initialize_centroids(X)
        prev_centroids = np.zeros_like(centroids)

        iter = -1
        for iter in range(self.max_iter):
            labels = self._assign_clusters(X, centroids)

            centroids = self._update_centeroids(X, labels)

            centroid_shift = np.sum((centroids - prev_centroids) ** 2)
            if centroid_shift < self.tol:
                break

            prev_centroids = centroids

        inertia = self._compute_inertia(X, labels, centroids)

        return KMeanResult(labels, centroids, iter + 1, inertia)

    def _initialize_centroids(
        self, X: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        indices = rng.permutation(n_samples)[: self.n_clusters]
        return X[indices].copy()

    def _assign_clusters(
        self, X: npt.NDArray[np.float64], centroids: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.int64]:
        """
        Assign each data point to nearest centroid.

        Args:
            X: Input data of shape (n_samples, n_features)
            centeroids: Current centeroids of shape (n_clusters, n_features)

        Returns:
            Array of labels of shape (n_samples,)
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.sum((X - centroids[k]) ** 2, axis=1)
        return np.argmin(distances, axis=1)

    def _update_centeroids(
        self, X: npt.NDArray[np.float64], labels: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """
        Update centeroids as mean of assigned points

        Args:
            X: Input data of shape (n_samples, n_features)
            labels: Cluster labels of shape (n_samples,)

        Returns:
            Updated centeroids of shape (n_clusters, n_features)
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                centroids[k] = np.mean(X[labels == k], axis=0)
        return centroids

    def _compute_inertia(
        self,
        X: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int64],
        centroids: npt.NDArray[np.float64],
    ) -> float:
        """
        Compute sum of squared distances of samples to their closest centroid.

        Args:
            X: Input data of shape (n_samples, n_features)
            labels: Cluster labels of shape (n_samples,)
            centroids: Current centeroids of shape (n_clusters, n_features)

        Returns:
            Inertia value
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            inertia += np.sum((X[labels == k] - centroids[k]) ** 2)
        return inertia


def main():
    data = load_iris()
    X, y = data.data, data.target

    kmeans = KMeans(n_clusters=3, random_state=0)
    result = kmeans.fit(X.astype(np.float64))

    n_features = X.shape[1]
    fig, axes = plt.subplots(n_features - 1, n_features - 1, figsize=(15, 15))

    for i in range(n_features - 1):
        for j in range(i + 1, n_features):
            ax = axes[i, j - 1]
            ax.scatter(X[:, i], X[:, j], c=result.labels, cmap="viridis")
            ax.scatter(
                result.centroids[:, i],
                result.centroids[:, j],
                c="red",
                marker="x",
                s=200,
                linewidths=3,
            )
            ax.set_xlabel(data.feature_names[i])
            ax.set_ylabel(data.feature_names[j])

    for i in range(n_features - 1):
        for j in range(n_features - 1):
            if j < i:
                fig.delaxes(axes[i][j])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
