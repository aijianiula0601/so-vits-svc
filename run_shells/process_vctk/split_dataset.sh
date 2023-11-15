#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../


out_dir=$1
train_file=$2
val_file=$3

python preprocess_flist_config.py \
  --train_list ${train_file} \
  --val_list ${val_file} \
  --source_dir ${out_dir} \
  --speech_encoder vec768l12 \
  --vol_aug
