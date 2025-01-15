from typing import TypeVar
import numpy as np
import numpy.typing as npt

from mllib.validation import check_consistent_length

T = TypeVar("T", bound=np.generic)
U = TypeVar("U", bound=np.generic)


def train_test_split(
    X: npt.NDArray[T], y: npt.NDArray[U], test_size: float, random_state: int | None
) -> tuple[npt.NDArray[T], npt.NDArray[T], npt.NDArray[U], npt.NDArray[U]]:
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data
    y : ndarray of shape (n_samples,)
        Target values
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int or None, default=None
        Controls the shuffling applied to the data before applying the split

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of ndarrays
        The split datasets
    """
    check_consistent_length(X, y)

    if not 0 <= test_size <= 1:
        raise ValueError(f"test_size must be between 0 and 1, got value {test_size}")

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    if random_state is not None:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    train_indices, test_indices = indices[n_test:], indices[:n_test]

    return (X[train_indices], X[test_indices], y[train_indices], y[test_indices])
