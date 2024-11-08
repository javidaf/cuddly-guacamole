import numpy as np
from ml_p1.utils import create_design_matrix
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    PowerTransformer,
)
from sklearn.model_selection import train_test_split
from ml_p1.functions import mse, r2_score
import matplotlib.pyplot as plt
from ml_p1.utils import generate_ff_data
from sklearn.linear_model import LinearRegression


def ols(X_train, z_train):
    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
    return beta


def ols_on_ff(
    xy: np.ndarray,
    z: np.ndarray,
    degree: int,
    scaling: str = "standard",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform OLS on Franke's function with polynomial features.

    Parameters:
    - xy: Input features.
    - z: Target values.
    - degree: Polynomial degree.
    - scaling: Scaling method ('standard', 'minmax', 'robust', 'power', or None).
    - random_state: Seed for reproducibility.

    Returns:
    - Tuple containing:
        - z_pred: Predictions on the entire dataset.
        - z_train_pred: Predictions on the training set.
        - z_test_pred: Predictions on the testing set.
        - z: Original target values.
        - z_train: Training target values.
        - z_test: Testing target values.
    - beta: OLS coefficients.
    - train_mse: Mean Squared Error on training set.
    - train_r2: R² score on training set.
    - test_mse: Mean Squared Error on testing set.
    - test_r2: R² score on testing set.
    """

    X = create_design_matrix(xy, degree=degree)

    X_train, X_test, z_train, z_test = train_test_split(
        X, z, test_size=0.2, random_state=random_state
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
    elif scaling is None:
        pass
    else:
        raise ValueError(f"Unsupported scaling method: {scaling}")

    if scaler_X and scaler_z:
        X_train = scaler_X.transform(X_train)
        X_test = scaler_X.transform(X_test)
        z_train = scaler_z.transform(z_train.reshape(-1, 1)).ravel()
        z_test = scaler_z.transform(z_test.reshape(-1, 1)).ravel()
        X = scaler_X.transform(X)

    beta = ols(X_train, z_train)

    z_train_pred = X_train @ beta
    z_test_pred = X_test @ beta
    z_pred = X @ beta

    z_pred = scaler_z.inverse_transform(z_pred.reshape(-1, 1)).ravel()

    train_mse = mse(z_train, z_train_pred)
    train_r2 = r2_score(z_train, z_train_pred)

    test_mse = mse(z_test, z_test_pred)
    test_r2 = r2_score(z_test, z_test_pred)

    return (
        (z_pred, z_train_pred, z_test_pred, z, z_train, z_test),
        beta,
        train_mse,
        train_r2,
        test_mse,
        test_r2,
    )


def ols_on_ff_evaluation(
    degrees: int = 5,
    scaling: str = "standard",
    noise: bool = True,
    n: int = 20,
    random_state: int = 42,
    xy=None,
    z=None,
):
    """
    Evaluate OLS on Franke's function across multiple polynomial degrees.

    Parameters:
    - degrees: Maximum polynomial degree to evaluate.
    - scaling: Scaling method ('standard', 'minmax', 'robust', 'power', or None).
    - noise: Whether to add noise to the Franke function data.
    - n: Number of data points to generate.
    - random_state: Seed for reproducibility.

    Returns:
    - mse_train: List of Mean Squared Errors on training sets for each degree.
    - r2_train: List of R² scores on training sets for each degree.
    - mse_test: List of Mean Squared Errors on testing sets for each degree.
    - r2_test: List of R² scores on testing sets for each degree.
    - betas: List of OLS coefficient arrays for each degree.
    """
    print(f"Running OLS on Franke's function up to degree {degrees}.")
    print(f"Scaling method: {scaling}")
    print(f"Adding noise: {noise}")
    print(f"Number of samples: {n}")

    # Initialize lists to store metrics and coefficients
    mse_train = []
    r2_train = []
    mse_test = []
    r2_test = []
    betas = []

    # Generate Franke's function data
    if xy is None or z is None:
        xy, z = generate_ff_data(n=n, noise=noise)

    for degree in range(1, degrees + 1):
        # Perform OLS on Franke's function for the current degree
        results = ols_on_ff(
            xy, z, degree=degree, scaling=scaling, random_state=random_state
        )

        (
            _,
            beta,
            train_mse,
            train_r2,
            test_mse,
            test_r2,
        ) = results

        # Append metrics and coefficients to respective lists
        mse_train.append(train_mse)
        r2_train.append(train_r2)
        mse_test.append(test_mse)
        r2_test.append(test_r2)
        betas.append(beta)

        # Optional: Print progress
    print(
        f"Max degree {degrees}: Train MSE={train_mse:.4f}, Train R²={train_r2:.4f}, "
        f"Test MSE={test_mse:.4f}, Test R²={test_r2:.4f}"
    )

    return mse_train, r2_train, mse_test, r2_test, betas


def visualize_betas_norm_ols(betas):
    # Calculate the norm of beta coefficients for each degree
    beta_norms = [np.linalg.norm(beta) for beta in betas]
    degrees = len(betas)
    # Plot the norm of beta coefficients against degree
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, degrees + 1), beta_norms, marker="o")
    plt.title("Norm of Beta Coefficients vs Polynomial Degree")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Norm of Beta Coefficients")
    plt.show()


def visualize_beta_histogram_ols(betas):
    degrees = len(betas)
    # Find the maximum number of beta coefficients across all degrees
    max_beta_length = max(len(beta) for beta in betas)

    # Initialize a matrix to hold beta coefficients
    beta_matrix = np.full((degrees, max_beta_length), np.nan)

    # Populate the beta matrix
    for i, beta in enumerate(betas):
        beta_matrix[i, : len(beta)] = beta

    # Plot the beta coefficients as a heatmap
    plt.figure(figsize=(8, 4))
    c = plt.imshow(
        beta_matrix.T, aspect="auto", cmap="viridis", interpolation="nearest"
    )
    plt.colorbar(c, label="Beta Coefficient Value")
    plt.title("Beta Coefficients vs Polynomial Degree")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Beta Coefficient Index")
    plt.gca().invert_yaxis()
    plt.xticks(ticks=range(degrees), labels=range(1, degrees + 1))

    # Add box outlines around zero beta coefficients
    ax = plt.gca()
    for (i, j), value in np.ndenumerate(beta_matrix.T):
        if np.isclose(value, 0):
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1
            )
            ax.add_patch(rect)

    plt.show()


def visualize_all_betas_ols(betas):
    # Determine the maximum number of beta coefficients across all degrees
    max_beta_length = max(len(beta) for beta in betas)

    plt.figure(figsize=(8, 4))
    for idx in range(max_beta_length):
        beta_values = [beta[idx] for beta in betas if len(beta) > idx]
        start_degree = next(i for i, beta in enumerate(betas) if len(beta) > idx) + 1
        plt.plot(range(start_degree, start_degree + len(beta_values)), beta_values)
    plt.title("All Beta Coefficients vs Polynomial Degree")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Beta Coefficient Value")
    plt.show()


def visualize_ols_metrics(mse_train, r2_train, mse_test, r2_test):
    degrees = len(mse_train)
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot MSE on the left y-axis
    ax1.plot(
        range(1, degrees + 1),
        mse_train,
        label="Train MSE",
        marker="x",
        color="tab:blue",
    )
    ax1.plot(
        range(1, degrees + 1),
        mse_test,
        label="Test MSE",
        marker="x",
        color="tab:orange",
    )
    ax1.set_xlabel("Polynomial Degree")
    ax1.set_ylabel("MSE", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    # Create a second y-axis for R2
    ax2 = ax1.twinx()
    ax2.plot(
        range(1, degrees + 1), r2_train, label="Train R2", marker="o", color="tab:green"
    )
    ax2.plot(
        range(1, degrees + 1), r2_test, label="Test R2", marker="o", color="tab:red"
    )
    ax2.set_ylabel("R2", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.legend(loc="upper right")

    plt.title("MSE and R2 vs Polynomial Degree")
    plt.tight_layout()
    plt.show()
