from sklearn.model_selection import train_test_split
from functions import mse, r2_score, FrankeFunction, Noise
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
)
from sklearn.linear_model import Lasso, Ridge
import numpy as np
from utils import create_design_matrix, generate_ff_data
import seaborn as sns


def lasso(X_train, z_train, alpha):
    lasso = Lasso(alpha=alpha, max_iter=5000)
    lasso.fit(X_train, z_train)

    beta = lasso.coef_
    return beta


def lasso_on_ff(
    xy: np.ndarray, z: np.ndarray, degree: int, alpha: float, scaling: str = "standard"
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    float,
    float,
    float,
    float,
]:
    X = create_design_matrix(xy, degree=degree)

    X_train, X_test, z_train, z_test = train_test_split(
        X, z, test_size=0.2, random_state=42
    )

    scaler_X = None
    scaler_z = None
    if scaling == "standard":
        scaler_X = StandardScaler().fit(X_train)
        scaler_z = StandardScaler().fit(z_train.reshape(-1, 1))
    elif scaling == "minmax":
        scaler_X = MinMaxScaler().fit(X_train)
        scaler_z = MinMaxScaler().fit(z_train.reshape(-1, 1))
    elif scaling == "robust":
        scaler_X = RobustScaler().fit(X_train)
        scaler_z = RobustScaler().fit(z_train.reshape(-1, 1))
    elif scaling == "power":
        scaler_X = PowerTransformer().fit(X_train)
        scaler_z = PowerTransformer().fit(z_train.reshape(-1, 1))

    if scaler_X and scaler_z:
        X_train = scaler_X.transform(X_train)
        X_test = scaler_X.transform(X_test)
        z_train = scaler_z.transform(z_train.reshape(-1, 1)).ravel()
        z_test = scaler_z.transform(z_test.reshape(-1, 1)).ravel()

    lasso = Lasso(alpha=alpha, max_iter=2000)
    lasso.fit(X_train, z_train)

    beta = lasso.coef_

    z_train_pred = lasso.predict(X_train)
    train_mse = mse(z_train, z_train_pred)
    train_r2 = r2_score(z_train, z_train_pred)

    z_test_pred = lasso.predict(X_test)
    test_mse = mse(z_test, z_test_pred)
    test_r2 = r2_score(z_test, z_test_pred)

    z_pred = lasso.predict(X)

    return (
        (z_pred, z_train_pred, z_test_pred, z, z_train, z_test),
        beta,
        train_mse,
        train_r2,
        test_mse,
        test_r2,
    )


def lasso_on_ff_evaluation(
    degrees: int = 5,
    scaling: str = "standard",
    noise: bool = True,
    n: int = 20,
    num_alphas: int = 4,
    xy: np.ndarray = None,
    z: np.ndarray = None,
) -> tuple[
    list[list[float]],
    list[list[float]],
    list[list[float]],
    list[list[float]],
    np.ndarray,
    list[np.ndarray],
]:
    mse_train = []
    mse_test = []
    r2_train = []
    r2_test = []
    betas = []
    alphas = np.logspace(-4, -1, num_alphas)
    if xy is None or z is None:
        xy, z = generate_ff_data(n=n, noise=noise)

    for alpha in alphas:
        mse_train_alpha = []
        mse_test_alpha = []
        r2_train_alpha = []
        r2_test_alpha = []
        betas_alpha = []

        for degree in range(1, degrees + 1):
            results = lasso_on_ff(xy, z, degree=degree, alpha=alpha, scaling=scaling)
            (
                _,
                beta,
                train_mse,
                train_r2,
                test_mse,
                test_r2,
            ) = results

            mse_train_alpha.append(train_mse)
            mse_test_alpha.append(test_mse)
            r2_train_alpha.append(train_r2)
            r2_test_alpha.append(test_r2)
            betas_alpha.append(beta)

        mse_train.append(mse_train_alpha)
        mse_test.append(mse_test_alpha)
        r2_train.append(r2_train_alpha)
        r2_test.append(r2_test_alpha)
        betas.append(betas_alpha)

    return mse_train, r2_train, mse_test, r2_test, alphas, betas


def visualize_lasso_betas_lasso(betas, alphas) -> None:
    max_beta_length = max(len(beta) for beta_set in betas for beta in beta_set)

    num_alphas = len(alphas)
    num_rows = (num_alphas + 1) // 2

    plt.figure(figsize=(12, 8))
    for i, alpha in enumerate(alphas):
        plt.subplot(num_rows, 2, i + 1)
        for idx in range(max_beta_length):
            beta_values = [beta[idx] for beta in betas[i] if len(beta) > idx]
            start_degree = (
                next(j for j, beta in enumerate(betas[i]) if len(beta) > idx) + 1
            )
            plt.plot(range(start_degree, start_degree + len(beta_values)), beta_values)
        if i >= num_alphas - 2:
            plt.xlabel("Polynomial Degree")
        plt.ylabel("Beta Coefficients")
        plt.title(f"Alpha: {alpha:.4f}")

    plt.tight_layout()
    plt.show()


def visualize_lasso_beta_histogram(betas, alphas):
    num_alphas = len(alphas)
    num_rows = (num_alphas + 1) // 2

    plt.figure(figsize=(12, 8))
    for i, (alpha, beta_set) in enumerate(zip(alphas, betas)):
        degrees = len(beta_set)
        max_beta_length = max(len(beta) for beta in beta_set)

        beta_matrix = np.full((degrees, max_beta_length), np.nan)

        for j, beta in enumerate(beta_set):
            beta_matrix[j, : len(beta)] = beta

        plt.subplot(num_rows, 2, i + 1)
        c = plt.imshow(
            beta_matrix.T,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            origin="lower",
        )
        plt.colorbar(c, label="Beta Coefficient Value")
        plt.title(f"Alpha: {alpha:.4f}")
        plt.xlabel("Polynomial Degree")
        plt.ylabel("Beta Coefficient Index")
        plt.xticks(ticks=range(degrees), labels=range(1, degrees + 1))

        ax = plt.gca()
        for (j, k), value in np.ndenumerate(beta_matrix.T):
            if np.isclose(value, 0):
                rect = plt.Rectangle(
                    (k - 0.5, j - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1
                )
                ax.add_patch(rect)

    plt.tight_layout()
    plt.show()


def visualize_lasso_betas_norm(betas, alphas):
    plt.figure(figsize=(10, 6))

    for alpha, beta_set in zip(alphas, betas):
        beta_norms = [np.linalg.norm(beta) for beta in beta_set]
        degrees = len(beta_set)
        plt.plot(
            range(1, degrees + 1), beta_norms, marker="o", label=f"Alpha: {alpha:.4f}"
        )

    plt.title("Norm of Beta Coefficients vs Polynomial Degree")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Norm of Beta Coefficients")
    plt.legend()
    plt.show()


def visualize_zero_coefficients(betas, alphas):
    plt.figure(figsize=(10, 6))

    for alpha, beta_set in zip(alphas, betas):
        zero_counts = [np.sum(np.isclose(beta, 0)) for beta in beta_set]
        total_counts = [len(beta) for beta in beta_set]
        zero_ratios = [zero / total for zero, total in zip(zero_counts, total_counts)]
        degrees = len(beta_set)
        plt.plot(
            range(1, degrees + 1), zero_ratios, marker="o", label=f"Alpha: {alpha:.4f}"
        )

    plt.title("Ratio of Zero Coefficients vs Polynomial Degree")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Ratio of Zero Coefficients")
    plt.legend()
    plt.show()
