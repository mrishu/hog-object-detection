import numpy as np
import cv2
import os
from tqdm import tqdm


def extract_negative_regions(image, labels, neg_size=(128, 256), max_attempts=50):
    """
    Extract a negative region from the image that does not overlap any given bounding boxes (labels).

    Parameters:
        - image (np.array): The input image.
        - labels (list of tuples):
            List of bounding boxes for humans, each in (class_id, center_x, center_y, width, height)
            normalized format.
        - neg_size (tuple): Desired size for the negative region as (width, height).
        - max_attempts (int): Number of attempts to find a valid negative region.

    Returns:
        np.array: The cropped negative region, or None if no valid region found.
    """
    img_height, img_width = image.shape[:2]
    neg_width, neg_height = neg_size

    # Ensure neg_size is feasible within the image dimensions
    if neg_width > img_width or neg_height > img_height:
        raise ValueError("Negative region size is larger than the input image size.")

    if labels.ndim == 1:
        labels = labels.reshape(-1, len(labels))
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

        # Check if the selected region overlaps any human bounding box
        overlaps = any(
            x_end > x_min and x_start < x_max and y_end > y_min and y_start < y_max
            for (x_min, y_min, x_max, y_max) in converted_labels
        )

        if not overlaps:
            # Extract the negative region from the image
            negative_region = image[
                y_start : y_start + neg_height, x_start : x_start + neg_width
            ]
            norm_neg_center_x = ((x_start + x_end) / 2.0) / img_width
            norm_neg_center_y = ((y_start + y_end) / 2.0) / img_height
            norm_neg_width = neg_width / img_width
            norm_neg_height = neg_height / img_height
            labels = np.array(
                [
                    1,
                    norm_neg_center_x,
                    norm_neg_center_y,
                    norm_neg_width,
                    norm_neg_height,
                ]
            )
            return negative_region, labels

    print("Could not find a valid negative region without overlap after max attempts.")
    return None, None


SPLIT = "valid"  # one of train, test or valid

image_dir = f"./inria/{SPLIT}/images/"
labels_dir = f"./inria/{SPLIT}/labels/"
neg_image_dir = f"./inria_neg/{SPLIT}/images"

dirs = [image_dir, labels_dir, neg_image_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

for file in tqdm(os.listdir(image_dir)):
    imfile = os.path.join(image_dir, file)
    image = cv2.imread(imfile).astype(np.float64)
    label_file = os.path.join(labels_dir, file[:-4] + ".txt")
    labels = np.loadtxt(label_file)
    neg_image, neg_label = extract_negative_regions(image, labels)
    i = 1
    while neg_label is not None:
        neg_image_filename = file[:-4] + f"-try{i}" + ".jpg"
        print(neg_image_filename)
        cv2.imwrite(
            os.path.join(neg_image_dir, neg_image_filename),
            neg_image,
        )
        labels = np.vstack((labels, neg_label))
        neg_image, neg_label = extract_negative_regions(image, labels)
        i += 1
