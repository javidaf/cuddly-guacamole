from sklearn.model_selection import train_test_split
from ml_p1.functions import mse, r2_score, FrankeFunction, Noise
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
)
from sklearn.linear_model import Ridge
import numpy as np
from ml_p1.utils import create_design_matrix, generate_ff_data


def ridge(X_train, z_train, alpha):
    return (
        np.linalg.inv(X_train.T @ X_train + alpha * np.eye(X_train.shape[1]))
        @ X_train.T
        @ z_train
    )


def ridge_on_ff(
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

    beta = ridge(X_train, z_train, alpha)

    z_train_pred = X_train @ beta
    train_mse = mse(z_train, z_train_pred)
    train_r2 = r2_score(z_train, z_train_pred)

    z_test_pred = X_test @ beta
    test_mse = mse(z_test, z_test_pred)
    test_r2 = r2_score(z_test, z_test_pred)

    z_pred = X @ beta

    return (
        (z_pred, z_train_pred, z_test_pred, z, z_train, z_test),
        beta,
        train_mse,
        train_r2,
        test_mse,
        test_r2,
    )


def ridge_on_ff_evaluation(
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
    alphas = np.logspace(-4, 1, num_alphas)

    if xy is None or z is None:
        xy, z = generate_ff_data(n=n, noise=noise)

    for alpha in alphas:
        alpha_mse_train = []
        alpha_mse_test = []
        alpha_r2_train = []
        alpha_r2_test = []
        alpha_betas = []

        for degree in range(1, degrees + 1):
            results = ridge_on_ff(xy, z, degree=degree, alpha=alpha, scaling=scaling)
            (
                _,
                beta,
                train_mse,
                train_r2,
                test_mse,
                test_r2,
            ) = results

            alpha_mse_train.append(train_mse)
            alpha_mse_test.append(test_mse)
            alpha_r2_train.append(train_r2)
            alpha_r2_test.append(test_r2)
            alpha_betas.append(beta)

        mse_train.append(alpha_mse_train)
        mse_test.append(alpha_mse_test)
        r2_train.append(alpha_r2_train)
        r2_test.append(alpha_r2_test)
        betas.append(alpha_betas)

    return mse_train, r2_train, mse_test, r2_test, alphas, betas


def visualize_ridge_on_ff_evaluation(
    mse_train, r2_train, mse_test, r2_test, alphas
) -> None:
    plt.figure(figsize=(12, 8))
    degrees = len(mse_train[0])
    num_alphas = len(alphas)
    num_rows = (num_alphas + 1) // 2

    for i, alpha in enumerate(alphas):
        plt.subplot(num_rows, 2, i + 1)
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        ax1.plot(
            range(1, degrees + 1),
            mse_train[i],
            label="Train MSE",
            marker="x",
            color="b",
        )
        ax1.plot(
            range(1, degrees + 1), mse_test[i], label="Test MSE", marker="x", color="g"
        )
        ax2.plot(
            range(1, degrees + 1), r2_train[i], label="Train R2", marker="o", color="r"
        )
        ax2.plot(
            range(1, degrees + 1), r2_test[i], label="Test R2", marker="o", color="m"
        )

        if i >= num_alphas - 2:
            ax1.set_xlabel("Polynomial Degree")

        ax1.set_ylabel("MSE", color="b")
        ax2.set_ylabel("R2", color="r")
        ax1.set_title(f"lambda: {alpha:.4f}")

        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def visualize_betas_ridge(betas, alphas) -> None:
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
        plt.title(f"Lambda: {alpha:.4f}")

    plt.tight_layout()
    plt.show()


def visualize_beta_histogram_ridge(betas, alphas):
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
            beta_matrix.T, aspect="auto", cmap="viridis", interpolation="nearest"
        )
        plt.colorbar(c, label="Beta Coefficient Value")
        plt.title(f"lambda: {alpha:.2f}")
        plt.xlabel("Polynomial Degree")
        plt.ylabel("Beta Coefficient Index")
        plt.xticks(ticks=range(degrees), labels=range(1, degrees + 1))

    plt.tight_layout()
    plt.show()


def visualize_betas_norm_ridge(betas, alphas):
    plt.figure(figsize=(10, 6))

    for alpha, beta_set in zip(alphas, betas):
        beta_norms = [np.linalg.norm(beta) for beta in beta_set]
        degrees = len(beta_set)
        plt.plot(
            range(1, degrees + 1), beta_norms, marker="o", label=f"lamda: {alpha:.4f}"
        )

    plt.title("Norm of Beta Coefficients vs Polynomial Degree")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Norm of Beta Coefficients")
    plt.legend()
    plt.show()
