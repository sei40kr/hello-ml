#!/usr/bin/env nix-shell
#!nix-shell -i python -p "python3.withPackages(ps: with ps; [ numpy matplotlib ])"

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class LogisticRegression:
    """
    Logistic Regression

    Prediction model:
    P(y=1|𝐱) = σ(α + 𝛃𝐱)

    Loss function (Binary Cross-Entropy):
    L(α,𝛃) = -(1/m) ∑ [y log(σ(α + 𝛃𝐱)) + (1 - y) log(1 - σ(α + 𝛃𝐱))]

    where:
    - σ(z) = 1 / (1 + exp(-z))
    - n: number of features
    - m: number of samples
    - X ∈ ℝᵐⁿ: input features
    - 𝐲 ∈ {0,1}ᵐ: binary target variable
    - α ∈ ℝ: bias
    - 𝛃 ∈ ℝⁿ: weights

    Gradients:
    - ∂L/∂α = (1/m) ∑ (σ(α + 𝛃𝐱) - y)
    - ∂L/∂𝛃 = (1/m) Xᵀ(σ(α + 𝛃X) - 𝐲)
    """

    η: float
    n_iterations: int
    α: float | None = None
    β: npt.NDArray[np.float64] | None = None

    def __init__(self, η: float = 0.01, n_iterations: int = 1_000):
        self.η = η
        self.n_iterations = n_iterations
        self.α = None
        self.β = None

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        m, n = X.shape

        α = 0.0
        β = np.zeros(n).astype(np.float64)

        for iteration in range(1, self.n_iterations + 1):
            z = α + np.dot(X, β)
            y_pred = self._sigmoid(z)

            dα = (1 / m) * np.sum(y_pred - y, dtype=np.float64)
            dβ = (1 / m) * np.dot(X.T, y_pred - y)

            α -= self.η * dα
            β -= self.η * dβ

            if iteration % 100 == 0:
                print(
                    f"Iteration {iteration}: BCE = {self._binary_cross_entropy(y_pred, y):.6f}, α = {α}, β = {β}"
                )

        self.α = α
        self.β = β

    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.α is None or self.β is None:
            raise RuntimeError("Model has not been trained yet")
        z = self.α + np.dot(X, self.β)
        return self._sigmoid(z)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (self.predict_proba(X) >= 0.5).astype(np.float64)

    def _sigmoid(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (1 / (1 + np.exp(-z))).astype(np.float64)

    def _binary_cross_entropy(
        self, y_pred: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64]
    ) -> float:
        return -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred),
            dtype=np.float64,
        )


def main():
    np.random.seed(0)

    α_true = 0.0
    β_true = 1.5

    X_data = (np.random.standard_normal(size=(100, 1)) * 2).astype(np.float64)
    z_data = (α_true + β_true * np.squeeze(X_data)).astype(np.float64)
    prob_data = (1 / (1 + np.exp(-z_data))).astype(np.float64)
    y_data = (np.random.rand(100) < prob_data).astype(np.float64)

    logistic_reg = LogisticRegression()
    logistic_reg.fit(X_data, y_data)

    X_test = np.reshape(np.linspace(X_data.min(), X_data.max(), 100), (-1, 1))
    prob_pred = logistic_reg.predict_proba(X_test)

    _, ax = plt.subplots(1, 1)

    ax.scatter(X_data, y_data, color="green", alpha=0.5, label="Data")
    ax.plot(X_test, prob_pred, color="red", label="Predicted Probability")

    ax.set_title("Logistic Regression")
    ax.legend()
    ax.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
