import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data_dir = "data"

X_test = np.loadtxt(os.path.join(data_dir, "test", "features.txt"))
y_test = np.loadtxt(os.path.join(data_dir, "test", "labels.txt"))

pipeline = pickle.load(open("model.pkl", "rb"))
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
