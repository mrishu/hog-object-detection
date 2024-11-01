import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle

data_dir = "data"

X_train = np.loadtxt(os.path.join(data_dir, "train", "features.txt"))
y_train = np.loadtxt(os.path.join(data_dir, "train", "labels.txt"))

# Create a SVM pipeline to scale and train
pipeline = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
pipeline.fit(X_train, y_train)

pickle.dump(pipeline, open("model.pkl", "wb"))
