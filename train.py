import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from vars import DATA_DIR

X_train = np.load(os.path.join(DATA_DIR, "train", "features.npy"))
y_train = np.load(os.path.join(DATA_DIR, "train", "labels.npy"))

print("\nTraining SVM...", end=" ")
# Create a SVM pipeline to scale and train
pipeline = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
pipeline.fit(X_train, y_train)
print("Done")

y_pred = pipeline.predict(X_train)

# Evaluate the model
accuracy = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred, average="binary")
recall = recall_score(y_train, y_pred, average="binary")
f1 = f1_score(y_train, y_pred, average="binary")

print("\nMetrics on Training Set: ")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

pickle.dump(pipeline, open("model.pkl", "wb"))
