import numpy as np
import cv2

import numpy as np


def cart_to_polar(x: np.ndarray, y: np.ndarray, angleInDegrees: bool = True):
    """Converts two arrays each in Cartesian coordinates to polar coordinates.

    Parameters:
    - x (np.ndarray): The x-coordinates.
    - y (np.ndarray): The y-coordinates.
    - angleInDegrees (bool): Whether to return the angle in degrees.

    Returns:
    - tuple[np.ndarray, np.ndarray]: The (magnitude, angle) tuple.
    """

    magnitude = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    if angleInDegrees:
        angle = np.degrees(angle)
    return magnitude, angle


def compute_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes an image and returns its gradient magnitude and angle.

    Parameters:
    - image (np.ndarray): The input image.

    Returns:
    - tuple[np.ndarray, np.ndarray]: The gradient (magnitude, angle) tuple.
    """
    dx_ker = np.array([[-1, 0, 1]], dtype=np.float64)
    dy_ker = dx_ker.T
    Ix = cv2.filter2D(image, -1, dx_ker)
    Iy = cv2.filter2D(image, -1, dy_ker)
    magnitude, angle = cart_to_polar(Ix, Iy, angleInDegrees=True)
    max_magnitude_idx = np.argmax(magnitude, axis=2)
    d1x, d2x = np.indices(max_magnitude_idx.shape)
    return magnitude[d1x, d2x, max_magnitude_idx], angle[d1x, d2x, max_magnitude_idx]
    # return magnitude, angle


def get_cell_descriptors(
    grad_magnitude: np.ndarray,
    grad_angle: np.ndarray,
    cell_size: tuple[int, int] = (8, 8),
    unsigned_grad: bool = True,
    num_bins: int = 9,
) -> np.ndarray:
    """
    Returns an array of HOG cell descriptor vectors for each cell in a window.

    Parameters:
    - grad_magnitude: The gradient magnitude of the window.
    - grad_angle: The gradient angle of the window (in degrees).
    - cell_size: The size of the cells in the window.
    - unsigned_grad: Whether the gradient angle is in the range [0, 180] or [0, 360].
    - num_bins: The number of histogram bins for each cell.

    Returns:
    - np.ndarray: An ndarray of shape (number of rows of cells, number of columns of cells, num_bins)
      containing the histogram of gradient for each cell.
    """

    if unsigned_grad:
        angle_range = (0, 180)
        grad_angle = np.mod(grad_angle, 180)
    else:
        angle_range = (0, 360)

    # Window dimensions
    shape_x = grad_magnitude.shape[0]
    shape_y = grad_magnitude.shape[1]

    num_cells_x = shape_x // cell_size[1]  # number of cells in x-direction
    num_cells_y = shape_y // cell_size[0]  # number of cells in y-direction

    # Initialize an empty list to collect histograms of each cell
    histograms = []

    # Loop through each cell in the window
    for x in range(num_cells_x):
        for y in range(num_cells_y):
            # Extract the cell's gradient magnitudes and angles
            cell_magnitudes = grad_magnitude[
                x * cell_size[1] : (x + 1) * cell_size[1],
                y * cell_size[0] : (y + 1) * cell_size[0],
            ]
            cell_angles = grad_angle[
                x * cell_size[1] : (x + 1) * cell_size[1],
                y * cell_size[0] : (y + 1) * cell_size[0],
            ]

            # Compute the histogram for the current cell
            hist, _ = np.histogram(
                cell_angles, bins=num_bins, range=angle_range, weights=cell_magnitudes
            )

            histograms.append(hist)

    return np.resize(
        np.array(histograms),
        (num_cells_x, num_cells_y, num_bins),
    )


def get_window_descriptor(
    grad_magnitude,
    grad_angle,
    cell_size=(8, 8),
    unsigned_grad=True,
    num_bins=9,
    block_size=(2, 2),
):
    """
    Parameters:
    - grad_magnitude: The gradient magnitude matrix of the window.
    - grad_angle: The gradient angle matrix of the window (in degrees).
    - cell_size: The size of the cells in the window.
    - unsigned_grad: Whether the gradient angle is in the range [0, 180] or [0, 360].
    - num_bins: The number of histogram bins for each cell.
    - block_size : number of cells in a block

    returns:
    - window_descriptor
    """
    cell_descriptors = get_cell_descriptors(
        grad_magnitude,
        grad_angle,
        cell_size=cell_size,
        unsigned_grad=unsigned_grad,
        num_bins=num_bins,
    )

    x_cells = cell_descriptors.shape[0]  # number of cells along x_axis
    y_cells = cell_descriptors.shape[1]  # number of cells along y_axis

    x_blocks = x_cells - block_size[0] + 1  # number of blocks along x_axis
    y_blocks = y_cells - block_size[1] + 1  # number of blocks along y_axis

    window_descriptors = []

    for i in range(x_blocks):
        for j in range(y_blocks):
            concat_vec = cell_descriptors[
                i : i + block_size[0], j : j + block_size[1]
            ].flatten()
            norm = np.linalg.norm(concat_vec)
            concat_vec = concat_vec / norm if norm != 0 else concat_vec
            window_descriptors.extend(concat_vec)

    return np.array(window_descriptors)
