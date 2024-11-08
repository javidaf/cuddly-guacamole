import numpy as np
from ml_p1.utils import create_design_matrix, generate_ff_data
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    PowerTransformer,
)
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ml_p1.part_a import ols


def bias_variance(
    degree=5,
    scaling="standard",
    noise=True,
    n=20,
    n_sim=100,
    xy=None,
    z=None,
    return_z_pred=False,
):

    if xy is None or z is None:
        xy, z = generate_ff_data(n=n, noise=noise)

    biases = []
    variances = []
    mses = []
    degrees = np.arange(1, degree + 1)

    for d in degrees:
        X = create_design_matrix(xy, degree=d)

        X_train, X_test, z_train, z_test = train_test_split(
            X, z, test_size=0.2, random_state=42
        )

        scaler_X = None
        scaler_z = None

        if scaling == "standard":
            scaler_X = StandardScaler().fit(X_train)
            X_train = scaler_X.transform(X_train)
            X_test = scaler_X.transform(X_test)

            scaler_z = StandardScaler().fit(z_train.reshape(-1, 1))
            z_train = scaler_z.transform(z_train.reshape(-1, 1)).ravel()
            z_test = scaler_z.transform(z_test.reshape(-1, 1)).ravel()
        elif scaling == "minmax":
            scaler_X = MinMaxScaler().fit(X_train)
            X_train = scaler_X.transform(X_train)
            X_test = scaler_X.transform(X_test)

            scaler_z = MinMaxScaler().fit(z_train.reshape(-1, 1))
            z_train = scaler_z.transform(z_train.reshape(-1, 1)).ravel()
            z_test = scaler_z.transform(z_test.reshape(-1, 1)).ravel()
        elif scaling == "robust":
            scaler_X = RobustScaler().fit(X_train)
            X_train = scaler_X.transform(X_train)
            X_test = scaler_X.transform(X_test)

            scaler_z = RobustScaler().fit(z_train.reshape(-1, 1))
            z_train = scaler_z.transform(z_train.reshape(-1, 1)).ravel()
            z_test = scaler_z.transform(z_test.reshape(-1, 1)).ravel()
        elif scaling == "power":
            scaler_X = PowerTransformer().fit(X_train)
            X_train = scaler_X.transform(X_train)
            X_test = scaler_X.transform(X_test)

            scaler_z = PowerTransformer().fit(z_train.reshape(-1, 1))
            z_train = scaler_z.transform(z_train.reshape(-1, 1)).ravel()
            z_test = scaler_z.transform(z_test.reshape(-1, 1)).ravel()
        else:
            # No scaling applied
            pass

        test_predictions = np.zeros((z_test.shape[0], n_sim))
        total_predictions = np.zeros((z.shape[0], n_sim))
        # total_predictions_test = np.zeros((Z_test.shape[0], n_sim))

        X_scaled = scaler_X.transform(X)
        for i in range(n_sim):

            X_resampled, z_resampled = resample(X_train, z_train, random_state=i)

            beta = ols(X_resampled, z_resampled)

            z_test_pred = X_test @ beta
            test_predictions[:, i] = z_test_pred.ravel()

            z_pred = X_scaled @ beta
            z_pred = scaler_z.inverse_transform(z_pred.reshape(-1, 1)).ravel()
            total_predictions[:, i] = z_pred

        z_test_pred_mean = np.mean(test_predictions, axis=1)
        z_pred_mean = np.mean(total_predictions, axis=1)
        bias_sq = np.mean((z_test - z_test_pred_mean) ** 2)
        biases.append(bias_sq)

        variance = np.mean(np.var(test_predictions, axis=1))
        variances.append(variance)

        mse_val = bias_sq + variance
        print(
            f"Degree {d}: Bias^2 = {bias_sq:.4f}, Variance = {variance:.4f}, MSE = {mse_val:.4f}"
        )
        mses.append(mse_val)

    if return_z_pred:
        return biases, variances, mses, z_pred_mean
    else:
        return biases, variances, mses


def visualize_bias_variance(biases, variances, mses):
    degrees = np.arange(1, len(biases) + 1)
    plt.figure()
    plt.plot(degrees, biases, label="Bias^2")
    plt.plot(degrees, variances, label="Variance")
    plt.plot(degrees, mses, label="MSE")
    plt.xlabel("Model Complexity (degree)")
    plt.ylabel("Error")
    plt.title("Bias-Variance Tradeoff")
    plt.legend()
    plt.show()
