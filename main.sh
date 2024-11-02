#!/bin/sh

rm -rf inria_neg data
python gen_neg.py
python prepare_data.py
python train.py
python valid.py
python test.py
