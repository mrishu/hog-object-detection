import cv2
import os
import numpy as np
from tqdm import tqdm

from hog import compute_gradients, get_window_descriptor
from utils import crop_image_using_labels

DETECTION_WIN_SIZE = (64, 128)  # in terms of pixels
CELL_SIZE = (8, 8)  # in terms of pixels
UNSIGNED_GRAD = True
NUM_BINS = 9
BLOCK_SIZE = (2, 2)  # in terms of number of cells


def get_hog_features(image_dir, labels_dir=None):
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


def prepare_data(pos_image_dir, neg_image_dir, pos_labels_dir=None):
    hog_features_pos = get_hog_features(pos_image_dir, pos_labels_dir)
    hog_features_neg = get_hog_features(neg_image_dir)

    y_pos = np.ones(len(hog_features_pos))
    y_neg = np.zeros(len(hog_features_neg))

    X = np.vstack((hog_features_pos, hog_features_neg))
    y = np.hstack((y_pos, y_neg))

    return X, y


data_dir = "data"

for split in ["train", "valid", "test"]:
    print("Preparing", split, "data")
    image_dir = os.path.join("inria", split, "images")
    label_dir = os.path.join("inria", split, "labels")
    image_dir_neg = os.path.join("inria_neg", split, "images")
    X, y = prepare_data(image_dir, image_dir_neg, label_dir)
    if not os.path.exists(os.path.join(data_dir, split)):
        os.makedirs(os.path.join(data_dir, split))
    np.savetxt(os.path.join(data_dir, split, "features.txt"), X)
    np.savetxt(os.path.join(data_dir, split, "labels.txt"), y)
