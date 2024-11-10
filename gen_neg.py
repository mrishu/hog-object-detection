import numpy as np
import cv2
import os
from tqdm import tqdm
from typing import Union

from vars import DATASET_DIR, NEG_DATASET_DIR

NEGATIVE_IMAGE_SIZE = (128, 256)

np.random.seed(100)


def extract_negative_regions(
    image: np.ndarray,
    labels: np.ndarray,
    neg_size: tuple[int, int] = NEGATIVE_IMAGE_SIZE,
    max_attempts: int = 500,
) -> Union[tuple[np.ndarray, np.ndarray], tuple[None, None]]:
    """
    Extract a negative region and its labels from the image that does not overlap any given bounding boxes (labels).

    Parameters:
        - image (np.ndarray): The input image.
        - labels (np.ndarray):
            Array of bounding boxes for humans, each row having [class_id, center_x, center_y, width, height]
            in normalized format.
        - neg_size (tuple[int, int]): Desired size for the negative region as (width, height).
        - max_attempts (int): Number of attempts to find a valid negative region.

    Returns:
        - tuple[np.array, np.array]: The (cropped negative region, label of negative region) tuple.
        - tuple[None, None]: if no valid region found.
    """
    img_height, img_width = image.shape[:2]
    neg_width, neg_height = neg_size

    # Ensure neg_size is feasible within the image dimensions
    if neg_width > img_width or neg_height > img_height:
        raise ValueError("Negative region size is larger than the input image size.")

    if labels.ndim == 1:
        labels = labels.reshape(-1, len(labels))

    # Unnormalize labels
    unnormalized_labels = [
        (
            int(center_x * img_width),
            int(center_y * img_height),
            int(width * img_width),
            int(height * img_height),
        )
        for _, center_x, center_y, width, height in labels
    ]

    # Convert center-based labels to corner-based bounding boxes
    converted_labels = [
        (
            int(center_x - width / 2),
            int(center_y - height / 2),
            int(center_x + width / 2),
            int(center_y + height / 2),
        )
        for center_x, center_y, width, height in unnormalized_labels
    ]

    # Attempt to find a valid negative region
    for _ in range(max_attempts):
        # Randomly select a top-left corner for the negative region
        x_start = np.random.randint(0, img_width - neg_width)
        y_start = np.random.randint(0, img_height - neg_height)

        # Define the bounding box for the negative region
        x_end = x_start + neg_width
        y_end = y_start + neg_height

        overlaps = any(
            x_end > x_min and x_start < x_max and y_end > y_min and y_start < y_max
            for (x_min, y_min, x_max, y_max) in converted_labels
        )

        if not overlaps:
            # Extract the negative region from the image
            negative_region = image[
                y_start : y_start + neg_height, x_start : x_start + neg_width
            ]
            # Normalized labels
            norm_neg_center_x = ((x_start + x_end) / 2.0) / img_width
            norm_neg_center_y = ((y_start + y_end) / 2.0) / img_height
            norm_neg_width = neg_width / img_width
            norm_neg_height = neg_height / img_height
            neg_label = np.array(
                [
                    1,
                    norm_neg_center_x,
                    norm_neg_center_y,
                    norm_neg_width,
                    norm_neg_height,
                ]
            )
            return negative_region, neg_label

    return None, None


scales = [0.4, 0.5, 0.8, 0.9, 1.0, 2.0]
# Looping over all the three splits
for split in ["train", "valid", "test"]:
    print(f"\nGenerating negative data for {split}...")
    image_dir = os.path.join(DATASET_DIR, split, "images")
    labels_dir = os.path.join(DATASET_DIR, split, "labels")
    neg_image_dir = os.path.join(NEG_DATASET_DIR, split, "images")

    dirs = [image_dir, labels_dir, neg_image_dir]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

    # Looping over every image file in each split
    for file in tqdm(os.listdir(image_dir)):
        imfile = os.path.join(image_dir, file)
        image = cv2.imread(imfile)
        label_file = os.path.join(labels_dir, file[:-4] + ".txt")
        labels = np.loadtxt(label_file)
        try_num = 1  # try number
        for scale in scales:
            neg_size = (
                int(NEGATIVE_IMAGE_SIZE[0] * scale),
                int(NEGATIVE_IMAGE_SIZE[1] * scale),
            )
            neg_image, neg_label = extract_negative_regions(
                image, labels, neg_size=neg_size
            )
            while neg_label is not None:
                neg_image_filename = file[:-4] + f"-try{try_num}" + ".jpg"
                cv2.imwrite(
                    os.path.join(neg_image_dir, neg_image_filename),
                    neg_image,
                )
                # Update labels array to include current extracted box so
                # that next extracted boxes doesn't overlap with current box as well
                labels = np.vstack((labels, neg_label))
                neg_image, neg_label = extract_negative_regions(image, labels)
                try_num += 1
