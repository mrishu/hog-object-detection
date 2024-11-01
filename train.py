"""Prepare training data for HOG model."""

from re import X
import cv2
import os
from utils import crop_image_using_labels
from hog import compute_gradients, get_window_descriptor
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline

DETECTION_WIN_SIZE = (64, 128)


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
            descriptor_vector = get_window_descriptor(grad_magnitude, grad_angle)
            hog_features.append(descriptor_vector)
    return hog_features


train_image_dir = "./inria/train/images/"
train_label_dir = "./inria/train/labels/"
hog_features_pos = get_hog_features(train_image_dir, train_label_dir)

train_image_dir_neg = "./inria_neg/train/images/"
hog_features_neg = get_hog_features(train_image_dir_neg)

# Labels for each class
y_pos = np.ones(len(hog_features_pos))
y_neg = np.zeros(len(hog_features_neg))

# Combine features and labels
X_train = np.vstack((hog_features_pos, hog_features_neg))
y_train = np.hstack((y_pos, y_neg))

# Create a pipeline with scaling and SVC
pipeline = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions
test_image_dir_pos = "./inria/test/images/"
test_label_dir_pos = "./inria/test/labels/"
hog_features_test_pos = get_hog_features(test_image_dir_pos, test_label_dir_pos)

test_image_dir_neg = "./inria_neg/test/images/"
hog_features_test_neg = get_hog_features(test_image_dir_neg)

X_test = np.vstack((hog_features_test_pos, hog_features_test_neg))

y_test_pos = np.ones(len(hog_features_test_pos))
y_test_neg = np.zeros(len(hog_features_test_neg))
y_test = np.hstack((y_test_pos, y_test_neg))

y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="binary")
recall = recall_score(y_test, y_pred, average="binary")
f1 = f1_score(y_test, y_pred, average="binary")

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
