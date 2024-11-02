#!/bin/sh

DATA_DIR="data"
NEG_DATASET_DIR="inria_neg"

rm -rf "$DATA_DIR" "$NEG_DATASET_DIR"

python gen_neg.py
python prepare_data.py
python train.py
python valid.py
python test.py
