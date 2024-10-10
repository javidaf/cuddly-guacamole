import numpy as np


def FrankeFunction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Franke's test function.
    ...

    Parameters
    ----------
    x : np.ndarray
        meshgrid x values.
    y : np.ndarray
        meshgrid y values.

    Returns
    -------
    np.ndarray
        meshgrid z values.
    """
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


def simple_1d_function(x):
    return 2 + 3 * x - 4 * x**2 + 5 * x**3


def Noise(s, n):
    return np.random.normal(0, s, n)


def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)


def msle(y, y_pred):
    return np.mean((np.log1p(y) - np.log1p(y_pred)) ** 2)


def r2_score(y, y_pred):
    """
    Calculate the R-squared score.

    Parameters:
    y (array-like): The true values.
    y_pred (array-like): The predicted values.

    Returns:
    float: The R-squared score.

    """
    return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)


def huber_c_function(r, delta):
    """
    Huber cost function.

    Parameters:
    r (array-like): The residuals.
    delta (float): The threshold.

    Returns:
    array-like: The cost.

    """
    return np.where(np.abs(r) <= delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta))


