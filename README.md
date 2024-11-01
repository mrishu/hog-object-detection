# hog-object-detection
Object Detection using HOG (Histogram of Gradients)

## Dataset
https://universe.roboflow.com/pascal-to-yolo-8yygq/inria-person-detection-dataset

Download the dataset, extract and rename it to `inria`. Place it in the same directory as the python files.

## How to execute
```bash
python gen_neg.py
python prepare_data.py
python train.py
python test.py
```

## Results
```
Accuracy: 0.97
Precision: 0.95
Recall: 0.96
F1 Score: 0.96
```
NOTE: Results may vary between different runs of the whole process.
