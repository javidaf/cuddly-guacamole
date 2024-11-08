import numpy as np
from utils import create_design_matrix, generate_ff_data
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from functions import mse  # Assuming this is your custom MSE function
import matplotlib.pyplot as plt
from part_e import bias_variance
from part_a import ols
from part_b import ridge
from part_c import lasso


def k_fold_cross_validation(
    xy: np.ndarray,
    z: np.ndarray,
    degree: int,
    scaling: str = "standard",
    k: int = 5,
    alphas: dict = {"ridge": 1.0, "lasso": 0.1},
    random_state: int = 42,
    return_z_pred: bool = False,
) -> dict:
    """
    Perform k-fold cross-validation for OLS, Ridge, and Lasso regression using custom functions.

    Parameters:
    - xy: Input features.
    - z: Target values.
    - degree: Polynomial degree for the design matrix.
    - scaling: Scaling method ('standard', 'minmax', 'robust', 'power', or None).
    - k: Number of folds for cross-validation.
    - alphas: Dictionary with alpha values for Ridge and Lasso.
    - random_state: Seed for reproducibility.

    Returns:
    - Dictionary with average MSE for each regression method.
    """

    X = create_design_matrix(xy, degree=degree)

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    mse_scores = {"OLS": [], "Ridge": [], "Lasso": []}
    predictions = {method: np.zeros_like(z) for method in mse_scores.keys()}

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"Processing Fold {fold}/{k}")

        X_train, X_test = X[train_index], X[test_index]
        z_train, z_test = z[train_index], z[test_index]

        # Initialize scalers
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
            X_train_scaled = scaler_X.transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            z_train_scaled = scaler_z.transform(z_train.reshape(-1, 1)).ravel()
            z_test_scaled = scaler_z.transform(z_test.reshape(-1, 1)).ravel()
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            z_train_scaled = z_train
            z_test_scaled = z_test

        # --- Ordinary Least Squares (OLS) ---
        beta_ols = ols(X_train_scaled, z_train_scaled)
        z_test_pred_ols = X_test_scaled @ beta_ols
        z_test_pred_ols_original = scaler_z.inverse_transform(
            z_test_pred_ols.reshape(-1, 1)
        ).ravel()
        predictions["OLS"][test_index] = z_test_pred_ols_original
        mse_ols = mse(z_test_scaled, z_test_pred_ols)
        mse_scores["OLS"].append(mse_ols)

        # --- Ridge Regression ---
        beta_ridge = ridge(
            X_train_scaled, z_train_scaled, alpha=alphas.get("ridge", 1.0)
        )
        z_test_pred_ridge = X_test_scaled @ beta_ridge
        z_test_pred_ridge_original = scaler_z.inverse_transform(
            z_test_pred_ridge.reshape(-1, 1)
        ).ravel()
        predictions["Ridge"][test_index] = z_test_pred_ridge_original
        mse_ridge = mse(z_test_scaled, z_test_pred_ridge)
        mse_scores["Ridge"].append(mse_ridge)

        # --- Lasso Regression ---
        beta_lasso = lasso(
            X_train_scaled, z_train_scaled, alpha=alphas.get("lasso", 0.1)
        )
        z_test_pred_lasso = X_test_scaled @ beta_lasso
        z_test_pred_lasso_original = scaler_z.inverse_transform(
            z_test_pred_lasso.reshape(-1, 1)
        ).ravel()
        predictions["Lasso"][test_index] = z_test_pred_lasso_original
        mse_lasso = mse(z_test_scaled, z_test_pred_lasso)
        mse_scores["Lasso"].append(mse_lasso)

    # Calculate average MSE for each method
    avg_mse_scores = {method: np.mean(scores) for method, scores in mse_scores.items()}

    # standard deviation
    std_mse_scores = {method: np.std(scores) for method, scores in mse_scores.items()}

    for method in avg_mse_scores:
        print(
            f"{method} Regression: Average MSE = {avg_mse_scores[method]:.4f} ± {std_mse_scores[method]:.4f}"
        )

    if return_z_pred:
        return {
            "average_mse": avg_mse_scores,
            "std_mse": std_mse_scores,
            "z_preds": predictions,
        }

    else:
        return {"average_mse": avg_mse_scores, "std_mse": std_mse_scores}


def compare_CV_bootstrap(
    degree: int = 5,
    scaling: str = "standard",
    noise: bool = True,
    n: int = 20,
    n_sim: int = 100,
    k: int = 10,
    alphas: dict = {"ridge": 1.0, "lasso": 0.00158},
    xy: np.ndarray = None,
    z: np.ndarray = None,
    return_z_pred: bool = False,
) -> None:

    z_pred = {"bootstrap_ols": None, "cv_ols": None, "cv_ridge": None, "cv_lasso": None}

    if xy is None or z is None:
        xy, z = generate_ff_data(n=n, noise=noise)

    # --- Bias-Variance Decomposition for OLS ---
    print("Running Bias-Variance Decomposition for OLS:")
    if return_z_pred:
        biases, variances, mses_bias_variance, z_pred_ols = bias_variance(
            degree=degree,
            scaling=scaling,
            noise=noise,
            n=n,
            n_sim=n_sim,
            xy=xy,
            z=z,
            return_z_pred=return_z_pred,
        )
        z_pred["bootstrap_ols"] = z_pred_ols
    else:
        biases, variances, mses_bias_variance = bias_variance(
            degree=degree,
            scaling=scaling,
            noise=noise,
            n=n,
            n_sim=n_sim,
            xy=xy,
            z=z,
            return_z_pred=return_z_pred,
        )

    # --- k-Fold Cross-Validation ---
    print("\nRunning k-Fold Cross-Validation:")
    mse_scores = {"ols": [], "ridge": [], "lasso": []}
    for d in range(1, degree + 1):
        if d == degree:
            results = k_fold_cross_validation(
                xy, z, degree=d, scaling=scaling, k=k, alphas=alphas, return_z_pred=True
            )
        else:
            results = k_fold_cross_validation(
                xy, z, degree=d, scaling=scaling, k=k, alphas=alphas
            )
        mse_scores["ols"].append(results["average_mse"]["OLS"])
        mse_scores["ridge"].append(results["average_mse"]["Ridge"])
        mse_scores["lasso"].append(results["average_mse"]["Lasso"])
    z_pred["cv_ols"] = results["z_preds"]["OLS"]
    z_pred["cv_ridge"] = results["z_preds"]["Ridge"]
    z_pred["cv_lasso"] = results["z_preds"]["Lasso"]
    # --- Comparison of MSEs ---
    degrees_range = np.arange(1, degree + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(degrees_range, mses_bias_variance, label="bootstrap MSE (OLS)", marker="o")
    plt.plot(
        degrees_range, mse_scores["ols"], label=f"{k}-Fold CV MSE (OLS)", marker="o"
    )
    plt.plot(
        degrees_range, mse_scores["ridge"], label=f"{k}-Fold CV MSE (Ridge)", marker="o"
    )
    plt.plot(
        degrees_range, mse_scores["lasso"], label=f"{k}-Fold CV MSE (Lasso)", marker="o"
    )

    min_index_bias_variance = mses_bias_variance.index(min(mses_bias_variance))
    min_index_ols = mse_scores["ols"].index(min(mse_scores["ols"]))
    min_index_ridge = mse_scores["ridge"].index(min(mse_scores["ridge"]))
    min_index_lasso = mse_scores["lasso"].index(min(mse_scores["lasso"]))

    plt.annotate(
        f"{min(mses_bias_variance):.3f}",
        (
            degrees_range[min_index_bias_variance],
            mses_bias_variance[min_index_bias_variance],
        ),
        textcoords="offset points",
        xytext=(0, 0),
        ha="center",
    )
    plt.annotate(
        f'{min(mse_scores["ols"]):.3f}',
        (degrees_range[min_index_ols], mse_scores["ols"][min_index_ols]),
        textcoords="offset points",
        xytext=(0, 0),
        ha="center",
    )
    plt.annotate(
        f'{min(mse_scores["ridge"]):.3f}',
        (degrees_range[min_index_ridge], mse_scores["ridge"][min_index_ridge]),
        textcoords="offset points",
        xytext=(0, 0),
        ha="center",
    )
    plt.annotate(
        f'{min(mse_scores["lasso"]):.3f}',
        (degrees_range[min_index_lasso], mse_scores["lasso"][min_index_lasso]),
        textcoords="offset points",
        xytext=(0, 0),
        ha="center",
    )

    plt.xlabel("Polynomial Degree")
    plt.ylabel("MSE")
    plt.title(
        "Comparison of MSE from Bias-Variance Decomposition and k-Fold Cross-Validation (OLS)"
    )
    plt.legend()
    plt.show()
    if return_z_pred:
        return z_pred


if __name__ == "__main__":
    compare_CV_bootstrap()
