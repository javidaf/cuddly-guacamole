from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler
from ml_p1.utils import create_design_matrix
from ml_p1.functions import mse, r2_score
from ml_p1.part_a import ols
from ml_p1.part_b import ridge
from ml_p1.part_c import lasso
import numpy as np
import matplotlib.pyplot as plt
import rasterio


import numpy as np
import rasterio


def read_and_preprocess_terrain(tif_file, subset_rows=None, subset_cols=None):
    with rasterio.open(tif_file) as dataset:
        z = dataset.read(1)
        transform = dataset.transform

    n_rows, n_cols = z.shape

    # Determine subset size
    if subset_rows is None or subset_cols is None:
        subset_rows, subset_cols = n_rows, n_cols

    # Calculate starting indices for the subset
    start_row = (n_rows - subset_rows) // 2
    start_col = (n_cols - subset_cols) // 2

    # Extract the subset
    z_subset = z[
        start_row : start_row + subset_rows, start_col : start_col + subset_cols
    ]

    rows, cols = np.meshgrid(
        np.arange(start_row, start_row + subset_rows),
        np.arange(start_col, start_col + subset_cols),
        indexing="ij",
    )
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs_flat = np.array(xs).flatten()
    ys_flat = np.array(ys).flatten()
    z_flat = z_subset.flatten()

    # Remove invalid data points
    valid_mask = ~np.isnan(z_flat)
    xs_flat = xs_flat[valid_mask]
    ys_flat = ys_flat[valid_mask]
    z_flat = z_flat[valid_mask]

    xy = np.column_stack((xs_flat, ys_flat))
    return xy, z_flat, valid_mask, z.shape


def evaluate_regression_methods(
    xy,
    z,
    degree=5,
    scaling="standard",
    lamda_ridge=1.0,
    alpha_lasso=0.00158,
    random_state=42,
):
    """
    Evaluate OLS, Ridge, and Lasso regression methods on the given dataset.

    Parameters:
    - xy: Input features (coordinates).
    - z: Target variable (elevation).
    - degree: Degree of the polynomial features.
    - scaling: Scaling method ('standard', 'minmax', 'robust', 'power', or None).
    - lamda_ridge: Regularization strength for Ridge regression.
    - alpha_lasso: Regularization strength for Lasso regression.
    - random_state: Seed for reproducibility.

    Returns:
    - results_dict: Dictionary containing predictions and metrics for each method.
    """
    # Create design matrix
    X = create_design_matrix(xy, degree=degree)

    # Split the data
    X_train, X_test, z_train, z_test = train_test_split(
        X, z, test_size=0.2, random_state=random_state
    )

    # Scaling
    scaler_X = None
    scaler_z = None

    if scaling == "standard":
        scaler_X = StandardScaler().fit(
            X_train[:, 1:],
        )  # Exclude intercept
        scaler_z = StandardScaler().fit(z_train.reshape(-1, 1))
    elif scaling == "minmax":
        scaler_X = MinMaxScaler().fit(X_train[:, 1:])
        scaler_z = MinMaxScaler().fit(z_train.reshape(-1, 1))
    elif scaling == "robust":
        scaler_X = RobustScaler().fit(X_train[:, 1:])
        scaler_z = RobustScaler().fit(z_train.reshape(-1, 1))
    elif scaling == "power":
        scaler_X = PowerTransformer().fit(X_train[:, 1:])
        scaler_z = PowerTransformer().fit(z_train.reshape(-1, 1))

    if scaler_X and scaler_z:
        # Scale only the feature columns (excluding the intercept term)
        X_train[:, 1:] = scaler_X.transform(X_train[:, 1:])
        X_test[:, 1:] = scaler_X.transform(X_test[:, 1:])
        X[:, 1:] = scaler_X.transform(X[:, 1:])
        z_train = scaler_z.transform(z_train.reshape(-1, 1)).ravel()
        z_test = scaler_z.transform(z_test.reshape(-1, 1)).ravel()
        z = scaler_z.transform(z.reshape(-1, 1)).ravel()

    results_dict = {}

    ## OLS Regression
    beta_ols = ols(X_train, z_train)
    z_train_pred_ols = X_train @ beta_ols
    z_test_pred_ols = X_test @ beta_ols
    z_pred_ols = X @ beta_ols

    train_mse_ols = mse(z_train, z_train_pred_ols)
    train_r2_ols = r2_score(z_train, z_train_pred_ols)
    test_mse_ols = mse(z_test, z_test_pred_ols)
    test_r2_ols = r2_score(z_test, z_test_pred_ols)

    if scaler_z:
        z_pred_ols = scaler_z.inverse_transform(z_pred_ols.reshape(-1, 1)).ravel()
        z_train_pred_ols = scaler_z.inverse_transform(
            z_train_pred_ols.reshape(-1, 1)
        ).ravel()
        z_test_pred_ols = scaler_z.inverse_transform(
            z_test_pred_ols.reshape(-1, 1)
        ).ravel()

    results_dict["OLS"] = {
        "beta": beta_ols,
        "z_pred": z_pred_ols,
        "train_mse": train_mse_ols,
        "train_r2": train_r2_ols,
        "test_mse": test_mse_ols,
        "test_r2": test_r2_ols,
    }

    ## Ridge Regression
    beta_ridge = ridge(X_train, z_train, alpha=lamda_ridge)
    z_train_pred_ridge = X_train @ beta_ridge
    z_test_pred_ridge = X_test @ beta_ridge
    z_pred_ridge = X @ beta_ridge

    train_mse_ridge = mse(z_train, z_train_pred_ridge)
    train_r2_ridge = r2_score(z_train, z_train_pred_ridge)
    test_mse_ridge = mse(z_test, z_test_pred_ridge)
    test_r2_ridge = r2_score(z_test, z_test_pred_ridge)

    if scaler_z:
        z_pred_ridge = scaler_z.inverse_transform(z_pred_ridge.reshape(-1, 1)).ravel()
        z_train_pred_ridge = scaler_z.inverse_transform(
            z_train_pred_ridge.reshape(-1, 1)
        ).ravel()
        z_test_pred_ridge = scaler_z.inverse_transform(
            z_test_pred_ridge.reshape(-1, 1)
        ).ravel()

    results_dict["Ridge"] = {
        "beta": beta_ridge,
        "z_pred": z_pred_ridge,
        "train_mse": train_mse_ridge,
        "train_r2": train_r2_ridge,
        "test_mse": test_mse_ridge,
        "test_r2": test_r2_ridge,
    }

    ## Lasso Regression
    beta_lasso = lasso(X_train, z_train, alpha=alpha_lasso)
    z_train_pred_lasso = X_train @ beta_lasso
    z_test_pred_lasso = X_test @ beta_lasso
    z_pred_lasso = X @ beta_lasso

    train_mse_lasso = mse(z_train, z_train_pred_lasso)
    train_r2_lasso = r2_score(z_train, z_train_pred_lasso)
    test_mse_lasso = mse(z_test, z_test_pred_lasso)
    test_r2_lasso = r2_score(z_test, z_test_pred_lasso)

    if scaler_z:
        z_pred_lasso = scaler_z.inverse_transform(z_pred_lasso.reshape(-1, 1)).ravel()
        z_train_pred_lasso = scaler_z.inverse_transform(
            z_train_pred_lasso.reshape(-1, 1)
        ).ravel()
        z_test_pred_lasso = scaler_z.inverse_transform(
            z_test_pred_lasso.reshape(-1, 1)
        ).ravel()

    results_dict["Lasso"] = {
        "beta": beta_lasso,
        "z_pred": z_pred_lasso,
        "train_mse": train_mse_lasso,
        "train_r2": train_r2_lasso,
        "test_mse": test_mse_lasso,
        "test_r2": test_r2_lasso,
    }

    return results_dict


def plot_terrain_predictions(results, valid_mask, z_shape, z_flat):
    methods = ["OLS", "Ridge", "Lasso"]
    n_methods = len(methods)
    plt.figure(figsize=(5 * n_methods, 6))

    # Original Terrain
    z_full = np.full(z_shape, np.nan)
    z_full.flat[valid_mask] = z_flat

    for i, method in enumerate(methods):
        z_pred_full = np.full(z_shape, np.nan)
        z_pred_full.flat[valid_mask] = results[method]["z_pred"]

        plt.subplot(1, n_methods, i + 1)
        plt.imshow(z_pred_full, cmap="terrain", interpolation="nearest")
        plt.title(f"{method} Predicted Terrain")
        plt.colorbar(label="Elevation")

    plt.tight_layout()
    plt.show()


def plot_3d_terrain_predictions(results, valid_mask, z_shape, z_flat):
    methods = ["OLS", "Ridge", "Lasso"]
    n_methods = len(methods)
    fig = plt.figure(figsize=(5 * n_methods, 6))

    # Original Terrain
    z_full = np.full(z_shape, np.nan)
    z_full.flat[valid_mask] = z_flat

    for i, method in enumerate(methods):
        z_pred_full = np.full(z_shape, np.nan)
        z_pred_full.flat[valid_mask] = results[method]["z_pred"]

        ax = fig.add_subplot(1, n_methods, i + 1, projection="3d")
        ax.plot_surface(
            *np.meshgrid(np.arange(z_shape[1]), np.arange(z_shape[0]), indexing="ij"),
            z_pred_full,
            cmap="terrain",
        )
        ax.set_title(f"{method} Predicted Terrain")
        ax.view_init(elev=30, azim=-90)  # Rotate the graph 90 degrees to the left

    plt.tight_layout()
    plt.show()


def plot_3d_terrain_predictions2(results, valid_mask, z_shape, z_flat, xy):
    methods = ["OLS", "Ridge", "Lasso"]
    n_methods = len(methods)
    fig = plt.figure(figsize=(5 * n_methods, 6))

    # Reshape x and y coordinates
    x = xy[:, 0].reshape(z_shape)
    y = xy[:, 1].reshape(z_shape)

    # Original Terrain
    z_full = np.full(z_shape, np.nan)
    z_full.flat[valid_mask] = z_flat

    for i, method in enumerate(methods):
        # Reshape predicted z values
        z_pred_full = np.full(z_shape, np.nan)
        z_pred_full.flat[valid_mask] = results[method]["z_pred"]

        ax = fig.add_subplot(1, n_methods, i + 1, projection="3d")
        ax.plot_surface(
            x,
            y,
            z_pred_full,
            cmap="terrain",
        )
        ax.set_title(f"{method} Predicted Terrain")
        ax.view_init(elev=30, azim=-90)  # Rotate the graph 90 degrees to the left

    plt.tight_layout()
    plt.show()
