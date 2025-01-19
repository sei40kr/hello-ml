import numpy as np
import pytest
from mllib.model_selection import train_test_split


def test_train_test_split():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int64)
    y = np.array([1, 2, 3, 4], dtype=np.int64)
    test_size = 0.25
    random_state = 42

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)

    assert X_train.shape == (3, 2)
    assert X_test.shape == (1, 2)
    assert y_train.shape == (3,)
    assert y_test.shape == (1,)

    # Check if the split is consistent
    assert set(y_train).union(set(y_test)) == set(y)
    assert set(map(tuple, X_train)).union(set(map(tuple, X_test))) == set(map(tuple, X))


def test_train_test_split_invalid_test_size():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int64)
    y = np.array([1, 2, 3, 4], dtype=np.int64)

    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=1.5, random_state=42)


def test_train_test_split_inconsistent_length():
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    y = np.array([1, 2, 3, 4], dtype=np.int64)

    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=0.25, random_state=42)
