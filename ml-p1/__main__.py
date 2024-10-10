from part_f import *
from part_g import read_and_preprocess_terrain
from plot import *
import argparse


def main(path, np, degree, scaling, lambda_ridge, alpha_lasso):
    tif_file = path
    subset_rows, subset_cols = np, np
    xy, z_flat, valid_mask, z_shape = read_and_preprocess_terrain(
        tif_file, subset_rows, subset_cols
    )
    z_reshaped_origin = z_flat.reshape((subset_rows, subset_cols))
    print("This from your data will be evaluated")

    plot_terrain_data(xy, z_reshaped_origin, subset_rows, subset_cols)

    z_pred = compare_CV_bootstrap(
        xy=xy,
        z=z_flat,
        degree=degree,
        scaling=scaling,
        alphas={"ridge": lambda_ridge, "lasso": alpha_lasso},
        return_z_pred=True,
    )

    z_ols_bootstrap = z_pred["cv_ols"].reshape((subset_rows, subset_cols))
    plot_terrain_data(
        xy=xy,
        z_reshaped_origin=z_ols_bootstrap,
        subset_cols=subset_cols,
        subset_rows=subset_rows,
    )
    z_ridge_bootstrap = z_pred["cv_ridge"].reshape((subset_rows, subset_cols))
    plot_terrain_data(
        xy=xy,
        z_reshaped_origin=z_ridge_bootstrap,
        subset_cols=subset_cols,
        subset_rows=subset_rows,
    )
    z_lasso_bootstrap = z_pred["cv_lasso"].reshape((subset_rows, subset_cols))
    plot_terrain_data(
        xy=xy,
        z_reshaped_origin=z_lasso_bootstrap,
        subset_cols=subset_cols,
        subset_rows=subset_rows,
    )
    z_ols_bootstrap = z_pred["cv_ols"].reshape((subset_rows, subset_cols))
    plot_terrain_data(
        xy=xy,
        z_reshaped_origin=z_ols_bootstrap,
        subset_cols=subset_cols,
        subset_rows=subset_rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process terrain data and plot results."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the terrain data file, should be tif file",
    )
    parser.add_argument(
        "--np",
        type=int,
        required=True,
        help="Number of points for subset rows and columns",
    )
    parser.add_argument(
        "--degree", type=int, required=True, help="Degree for polynomial fitting"
    )
    parser.add_argument(
        "--scaling",
        type=str,
        required=True,
        choices=["standard", "none"],
        help="Scaling method",
    )
    parser.add_argument(
        "--lambda_ridge",
        type=float,
        required=True,
        help="Lambda value for ridge regression",
    )
    parser.add_argument(
        "--alpha_lasso",
        type=float,
        required=True,
        help="Alpha value for lasso regression",
    )

    args = parser.parse_args()
    main(
        args.path,
        args.np,
        args.degree,
        args.scaling,
        args.lambda_ridge,
        args.alpha_lasso,
    )
