import numpy as np
import cv2
import matplotlib.pyplot as plt
import hog

DETECTION_WINDOW_SIZE = (128, 64)  # in terms of pixels
CELL_SIZE = (6, 6)  # in terms of pixels
BLOCK_SIZE = (3, 3)  # in terms of cells
UNSIGNED_GRADIENT = True  # whether to use 0-180 or 0-360 for gradient angle
NUM_ORIENTATION_BINS = 9  # number of bins in orientation histogram


## TODO: change this with actual image
image = np.zeros(DETECTION_WINDOW_SIZE)

grad_magnitude, grad_angle = hog.compute_gradients(image)
cell_descriptors = hog.get_cell_descriptors(
    grad_magnitude,
    grad_angle,
    cell_size=CELL_SIZE,
    unsigned_grad=UNSIGNED_GRADIENT,
    num_bins=NUM_ORIENTATION_BINS,
)
