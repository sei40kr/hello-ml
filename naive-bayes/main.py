#!/usr/bin/env nix-shell
#!nix-shell -i python -p "python3.withPackages(ps: with ps; [ matplotlib numpy pandas ])"

from math import exp
from typing import Generic, TypeVar
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt

FeatureType = TypeVar("FeatureType", bound=np.generic)
TargetType = TypeVar("TargetType", bound=np.generic)


class NaiveBayesClassifier(Generic[FeatureType, TargetType]):
    """
    Naive Bayes classifier

    Naive Bayes Theory:
    P(y|x) ∝ P(x|y)P(y)

    Assuming conditional independence of features:
    P(x|y) = ΠP(xᵢ|y)

    Using log-likehood for numerical stability:
    log P(y|x) = log P(y) + ∑log P(xᵢ|y)
    """

    class_prior: dict[TargetType, float]
    feature_probs: dict[TargetType, dict[int, dict[FeatureType, float]]]

    def __init__(self):
        self.class_priors = {}
        self.feature_probs = {}

    def fit(self, X: npt.NDArray[FeatureType], y: npt.NDArray[TargetType]) -> None:
        """
        Train the model

        Parameters:
            X: Feature matrix of shape (n, m)
            y: Target array of shape (n)
        where:
            n: Number of samples
            m: Number of features
        """
        n_samples = len(y)

        unique_classes = np.unique(y)
        for cls in unique_classes:
            self.class_priors[cls] = np.sum(y == cls) / n_samples

        for cls in unique_classes:
            self.feature_probs[cls] = {}
            cls_mask = y == cls

            for feature_idx in range(X.shape[1]):
                feature_values, counts = np.unique(
                    X[cls_mask, feature_idx], return_counts=True
                )

                # Calculate probabilities with Laplace smoothing
                self.feature_probs[cls][feature_idx] = {}
                n_values = len(feature_values)
                for feature_value, count in zip(feature_values, counts):
                    self.feature_probs[cls][feature_idx][feature_value] = (
                        count + 1
                    ) / (np.sum(cls_mask) + n_values)

    def predict_proba(self, x: npt.NDArray[FeatureType]) -> dict[TargetType, float]:
        """Calculate posterior probabilities for each class

        Args:
            x: Feature vector to predict of shape (m)
        where:
            m: Number of features

        Returns:
            Prediction probabilities for each class
        """
        log_probs: dict[TargetType, float] = {}

        for cls in self.class_priors:
            log_prob = np.log(self.class_priors[cls])

            for feature_idx, feature_val in enumerate(x):
                if feature_val in self.feature_probs[cls][feature_idx]:
                    log_prob += np.log(
                        self.feature_probs[cls][feature_idx][feature_val]
                    )
                else:
                    # REVIEW: Use Laplace smoothing value for unknown feature values
                    n_values = len(self.feature_probs[cls][feature_idx])
                    log_prob += np.log(
                        1 / (len(self.feature_probs[cls][feature_idx]) + n_values)
                    )

            log_probs[cls] = log_prob

        return self._softmax(log_probs)

    def predict(self, x: npt.NDArray[FeatureType]) -> TargetType:
        """Predict the most probable class

        Args:
            x: Feature vector to predict of shape (m)
        where:
            m: Number of features

        Returns:
            Predicted class
        """
        probs = self.predict_proba(x)
        return max(probs.items(), key=lambda x: x[1])[0]

    def _softmax(self, log_probs: dict[TargetType, float]) -> dict[TargetType, float]:
        """Apply softmax function to log probabilities

        Softmax function:
        σ(𝐳)ᵢ = exp(𝐳ᵢ) / ∑ exp(z)

        Args:
            log_probs: Dictionary of log probabilities

        Returns:
            Dictionary of normalized probabilities
        """
        max_log_prob = max(log_probs.values())
        probs = {
            cls: exp(log_prob - max_log_prob) for cls, log_prob in log_probs.items()
        }
        prob_sum = sum(probs.values())
        return {cls: prob / prob_sum for cls, prob in probs.items()}


def main():
    df = pd.read_csv("../datasets/iris/iris.csv")
    df = df.sample(frac=1, random_state=0)

    X = df.drop("species", axis=1).values.astype(np.float64)
    y = df["species"].values.astype(np.str_)

    test_size = 0.3
    test_count = int(len(df) * test_size)

    X_train, X_test = X[:-test_count], X[-test_count:]
    y_train, y_true = y[:-test_count], y[-test_count:]

    classifier = NaiveBayesClassifier()
    classifier.fit(X_train, y_train)

    y_pred = np.array([classifier.predict(x) for x in X_test])

    classes = sorted(set(y_true))
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        cm[classes.index(true), classes.index(pred)] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
