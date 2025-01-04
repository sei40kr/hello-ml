#!/usr/bin/env nix-shell
#!nix-shell -i python -p "python3.withPackages(ps: with ps; [ numpy matplotlib ])"

from typing import cast
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class LinearRegression:
    """
    Prediction model:
    𝐲 = 𝑿𝛃 + ε

    Loss function (Mean Squared Error):
    L(𝛃, ε) = ∑ (y_pred - y)² / m
            = ∑ (𝐱𝛃 + ε - y)² / m

    where:
    - n: number of features
    - m: number of samples
    - 𝑿 ∈ ℝᵐⁿ: input features
    - 𝐲 ∈ ℝᵐ: target variable
    - 𝛃 ∈ ℝⁿ: weights
    - ε ∈ ℝ: bias
    """

    η: float
    n_iterations: int
    β: npt.NDArray[np.float64] | None
    ε: float | None

    def __init__(self, η: float = 0.01, n_iterations: int = 1000):
        self.η = η
        self.n_iterations = n_iterations
        self.β = None
        self.ε = None

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        m, n = X.shape

        β = np.zeros(n)
        ε = 0.0

        for iteration in range(self.n_iterations):
            y_pred: float = np.dot(X, β) + ε

            dβ = (1 / m) * np.dot(X.T, y_pred - y)
            dε = (1 / m) * cast(float, np.sum(y_pred - y))

            β -= self.η * dβ
            ε -= self.η * dε

            if iteration % 100 == 0:
                mse = np.mean((y_pred - y) ** 2)
                print(f"Iteration {iteration}: MSE = {mse:.6f}, β = {β}, ε = {ε}")

        self.β = β
        self.ε = ε

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.β is None or self.ε is None:
            raise ValueError("Model has been not trained yet")
        return np.dot(X, self.β) + self.ε


def main():
    np.random.seed(0)

    β_true = 2.0
    ε_true = 1.0

    X_data = np.random.randn(100, 1)
    y_data = cast(
        npt.NDArray[np.float64],
        β_true * np.squeeze(X_data) + ε_true + 0.1 * np.random.randn(100),
    )

    linear_reg = LinearRegression()
    linear_reg.fit(X=X_data, y=y_data)

    X_pred = cast(
        npt.NDArray[np.float64],
        np.reshape(
            np.linspace(start=X_data.min(), stop=X_data.max(), num=100),
            (-1, 1),
        ),
    )
    y_pred = linear_reg.predict(X=X_pred)

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))

    ax.set_title(label="Linear Regression")
    ax.grid(visible=True)

    ax.scatter(X_data, y_data, color="blue", alpha=0.5, label="Data")
    ax.plot(X_pred, y_pred, color="red", label="Prediction")

    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
