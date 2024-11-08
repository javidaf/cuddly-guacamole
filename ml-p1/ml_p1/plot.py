from functions import FrankeFunction, simple_1d_function
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def plot_2d_function(x, y, z):
    """
    PLot 2D function.

    Parameters
    ----------
    x : np.ndarray
        x values. shape (n, m) meshgrid.
    y : np.ndarray
        y values. shape (n, m) meshgrid.
    z : np.ndarray
        z values. shape (n, m) meshgrid.

    Returns
    -------
    None

    """

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.0, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_1d_function(x, y):
    plt.plot(x, y)
    plt.show()


def visualize_betas_norm(betas):
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


def visualize_beta_histogram(betas):
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
    plt.xticks(ticks=range(degrees), labels=range(1, degrees + 1))
    plt.show()


def visualize_all_betas(betas):
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


def plot_terrain_data(xy, z_reshaped_origin, subset_rows, subset_cols,name="Original Terrain Data"):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # Plot the terrain data as a 2D image with a grayscale colormap
    axs[0].imshow(z_reshaped_origin, cmap='terrain')
    axs[0].set_title('name')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    fig.colorbar(axs[0].imshow(z_reshaped_origin, cmap='terrain'), ax=axs[0], label='Elevation')

    # Create a 3D plot of the terrain data
    ax = fig.add_subplot(122, projection='3d')
    x = xy[:, 0].reshape((subset_rows, subset_cols))
    y = xy[:, 1].reshape((subset_rows, subset_cols))
    ax.plot_surface(x, y, z_reshaped_origin, cmap='terrain')
    ax.view_init(elev=60, azim=180)  # Rotate the graph 90 degrees to the left

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Elevation')
    ax.set_title(name)

    plt.show()