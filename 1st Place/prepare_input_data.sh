#!/usr/bin/env bash

set -e

# uncompress original dataset to input/raw/ directory
# decoded frames are stored to /opt/data_fast/pri_matrix/train_img/
# ssd drive is very recommended with ~200-300GB free space

cd input
mkdir -p raw_test
mkdir -p raw_unused
mkdir -p /opt/data_fast/pri_matrix/train_img/


pushd ../src
echo "split dataset to raw, raw_test and raw_unused"
python generate_folds.py
popd


cd raw
bash ../convert_all_img.sh

