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
