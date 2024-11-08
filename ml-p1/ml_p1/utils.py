import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from ml_p1.functions import FrankeFunction, Noise
from matplotlib import pyplot as plt


def create_design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Create a design matrix with polynomial features.
    ...

    Parameters
    ----------
    x : np.ndarray
        Input data, with shape (n, m), n: #samples and m: #features.
    degree : int
        The degree of the polynomial features.


    Returns
    -------
    np.ndarray
        The return value. True for success, False otherwise.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=True)

    X = poly.fit_transform(x)
    # print("[CREATE DESIGN MATRIX] X shape:", X.shape)
    return X


def generate_ff_data(n: int = 20, noise: bool = False):
    """
    Generate data for Franke's function.
    ...

    Parameters
    ----------
    n : int
        Number of samples.
    noise : bool, optional
        Add noise to the data, by default False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        xy = [[x1  y1] [x2  y2] [x3  y3] ...] shape (n, 2).
        z = meshgrid z values.
    """
    # x = np.arange(0, 1, 1 / n)
    # y = np.arange(0, 1, 1 / n)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    # xy = np.concatenate(
    #     (x.reshape(-1, 1), y.reshape(-1, 1)), axis=1
    # )  # [[x1  y1] [x2  y2] [x3  y3] ...]

    x_mesh, y_mesh = np.meshgrid(x, y)
    xy = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])  # Shape: (n*n, 2)
    if noise:
        z = FrankeFunction(x_mesh, y_mesh) + Noise(0.1, x_mesh.shape)
    else:
        z = FrankeFunction(x_mesh, y_mesh)

    z = z.ravel()
    # print("[GENERATE FF DATA] xy shape:", xy.shape)
    # print("[GENERATE FF DATA] z shape:", z.shape)

    return xy, z


def visualize_evaluation_metrics(
    mse_train, r2_train, mse_test, r2_test, alphas
) -> None:

    plt.figure(figsize=(12, 8))
    degrees = len(mse_train[0])
    num_alphas = len(alphas)
    num_rows = (
        num_alphas + 1
    ) // 2  # Calculate the number of rows needed for two columns

    for i, alpha in enumerate(alphas):
        plt.subplot(num_rows, 2, i + 1)
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        ax1.plot(
            range(1, degrees + 1),
            mse_train[i],
            label="Train MSE",
            color="b",
            linestyle="--",
        )
        ax1.plot(
            range(1, degrees + 1),
            mse_test[i],
            label="Test MSE",
            color="orange",
            linestyle="--",
        )
        ax2.plot(range(1, degrees + 1), r2_train[i], label="Train R2", color="g")
        ax2.plot(range(1, degrees + 1), r2_test[i], label="Test R2", color="r")

        if i >= num_alphas - 2:
            ax1.set_xlabel("Polynomial Degree")

        ax1.set_ylabel("MSE", color="b")
        ax2.set_ylabel("R2", color="r")
        ax1.set_title(f"lambda: {alpha:.5f}")

        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def visualize_betas(betas, alphas) -> None:
    plt.figure(figsize=(12, 8))
    degrees = len(betas[0])
    for i, alpha in enumerate(alphas):
        plt.subplot(len(alphas), 1, i + 1)
        for j in range(len(betas[0][0])):
            plt.plot(range(1, degrees + 1), [beta[j] for beta in betas[i]])
        if i == len(alphas) - 1:
            plt.xlabel("Polynomial Degree")
        plt.ylabel("Beta Coefficients")
        plt.title(f"Alpha: {alpha:.2f}")

    plt.tight_layout()
    plt.show()
