import cv2
import os
import numpy as np
from tqdm import tqdm
from typing import Union

from hog import compute_gradients, get_window_descriptor
from utils import crop_image_using_labels

from vars import (
    DATASET_DIR,
    NEG_DATASET_DIR,
    NUM_BINS,
    BLOCK_SIZE,
    CELL_SIZE,
    UNSIGNED_GRAD,
    DETECTION_WIN_SIZE,
)


def get_hog_features(
    image_dir: str, labels_dir: Union[str, None] = None
) -> list[np.ndarray]:
    """
    Takes an image directory and it's corresponding labels directory (optional)
    and returns a list of HOG features for each image.

    Parameters:
    - image_dir: The directory of the images.
    - labels_dir: The directory of the labels (optional).

    Returns:
    - list[np.ndarray]: A list of HOG features for each image.
    """
    hog_features = []
    for file in tqdm(os.listdir(image_dir)):
        imfile = os.path.join(image_dir, file)
        image = cv2.imread(imfile).astype(np.float64)
        if labels_dir:
            label_file = os.path.join(labels_dir, file[:-4] + ".txt")
            labels = np.loadtxt(label_file)
            cropped_images = crop_image_using_labels(image, labels)
        else:
            cropped_images = [image]
        for cropped_image in cropped_images:
            cropped_image = cv2.resize(
                cropped_image, DETECTION_WIN_SIZE, cv2.INTER_AREA
            )
            grad_magnitude, grad_angle = compute_gradients(cropped_image)
            descriptor_vector = get_window_descriptor(
                grad_magnitude,
                grad_angle,
                cell_size=CELL_SIZE,
                unsigned_grad=UNSIGNED_GRAD,
                num_bins=NUM_BINS,
                block_size=BLOCK_SIZE,
            )
            hog_features.append(descriptor_vector)
    return hog_features


data_dir = "data"

# Loop for each split
for split in ["train", "valid", "test"]:
    print(f"\nPreparing data for {split}...")

    # Relevant Directories
    pos_image_dir = os.path.join(DATASET_DIR, split, "images")
    pos_labels_dir = os.path.join(DATASET_DIR, split, "labels")
    neg_image_dir = os.path.join(NEG_DATASET_DIR, split, "images")

    # Prepare data
    hog_features_pos = get_hog_features(pos_image_dir, pos_labels_dir)
    hog_features_neg = get_hog_features(neg_image_dir)
    X = np.vstack((hog_features_pos, hog_features_neg))
    y_pos = np.ones(len(hog_features_pos))
    y_neg = np.zeros(len(hog_features_neg))
    y = np.hstack((y_pos, y_neg))

    if not os.path.exists(os.path.join(data_dir, split)):
        os.makedirs(os.path.join(data_dir, split), exist_ok=True)
    np.save(os.path.join(data_dir, split, "features.npy"), X)
    np.save(os.path.join(data_dir, split, "labels.npy"), y)
