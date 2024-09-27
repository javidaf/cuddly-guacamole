from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from functions import FrankeFunction, noise, mse, r2_score
from plot import plot_2d_function
from sklearn.model_selection import train_test_split
from functions import simple_1d_function
from plot import plot_1d_function
from sklearn.preprocessing import StandardScaler


def create_design_matrix(x, d, y=None):
    """
    Create a design matrix for polynomial regression.
    Parameters:
    x (ndarray): Input array of shape (N,) or (N, M).
    y (ndarray): Input array of shape (N,) or (N, M). Default is None.
    d (int): Degree of the polynomial.
    Returns:
    ndarray: Design matrix of shape (N, l), where l is the number of features in the design matrix.
    Raises:
    None
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        if y is not None:
            y = np.ravel(y)

    N = len(x)
    l = int((d + 1) * (d + 2) / 2)
    X = np.ones((N, l))

    for i in range(1, d + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            if y is not None:
                X[:, q + k] = (x ** (i - k)) * (y**k)
            else:
                X[:, q + k] = x ** (i - k)

    return X


# make data

# x = np.arange(0, 1, 0.05)
# y = np.arange(0, 1, 0.05)
# x, y = np.meshgrid(x, y)

# z = FrankeFunction(x, y) + noise(0.01, x.shape)


# plot_2d_function(x, y, z)
def main():
    # Generate grid
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    z = FrankeFunction(x, y)

    z = np.ravel(z)

    X = create_design_matrix(x, 3, y)

    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)

    scaler_z = StandardScaler()
    z = scaler_z.fit_transform(z.reshape(-1, 1)).ravel()

    X_train, X_test, z_train, z_test = train_test_split(
        X, z, test_size=0.2, random_state=42
    )

 
    beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train

    z_train_pred = X_train @ beta
    print("Train MSE:", mse(z_train, z_train_pred))
    print("Train R2 score:", r2_score(z_train, z_train_pred))

    z_test_pred = X_test @ beta
    print("Test MSE:", mse(z_test, z_test_pred))
    print("Test R2 score:", r2_score(z_test, z_test_pred))

    z_pred = (X @ beta).reshape(x.shape)

    z_pred = scaler_z.inverse_transform(z_pred)

    plot_2d_function(x, y, z_pred)


if __name__ == "__main__":
    main()
