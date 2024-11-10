# hog-object-detection
Object Detection using HOG (Histogram of Gradients).

## Dataset
https://universe.roboflow.com/pascal-to-yolo-8yygq/inria-person-detection-dataset/dataset/1

Download the dataset, extract and rename it to `inria`. Place it in the same directory as the python files.

## Paper
https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf

## How to execute
1. Download the dataset and place it in the repository directory with name `inria`.
2. Generate negative examples with `python gen_neg.py`.
3. Prepare and save HOG features for all splits of data with `python prepare_data.py`.
4. Open `hog-human-detection.ipynb` in Jupyter and execute cells to train and run classifier on testing and validation data. Also the last cell executes sliding windows to find humans in random images picked from test set.

## Results
```
Metrics on Validation Set: 
Accuracy: 0.95
Precision: 0.96
Recall: 0.93
F1 Score: 0.95

Metrics on Test Set:
Accuracy: 0.96
Precision: 0.95
Recall: 0.95
F1 Score: 0.95
```
