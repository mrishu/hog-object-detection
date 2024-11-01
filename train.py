import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline

# Read saved data from data_dir
data_dir = "./data/"

X_train = np.loadtxt(os.path.join(data_dir, "train", "features.txt"))
y_train = np.loadtxt(os.path.join(data_dir, "train", "labels.txt"))

# Create a SVM pipeline to scale and train
pipeline = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
pipeline.fit(X_train, y_train)

# Make predictions using trained model
X_test = np.loadtxt(os.path.join(data_dir, "test", "features.txt"))
y_test = np.loadtxt(os.path.join(data_dir, "test", "labels.txt"))

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
