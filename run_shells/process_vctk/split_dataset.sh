#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../


base_dir="/mnt/cephfs/hjh/train_record/vc/so-vits-svc/vctk/test_data"
out_dir="${base_dir}/resample_to_44100"
train_file="${base_dir}/train.txt"
val_file="${base_dir}/val.txt"

python preprocess_flist_config.py \
  --train_list ${train_file} \
  --val_list ${val_file} \
  --source_dir ${out_dir} \
  --speech_encoder vec768l12 \
  --vol_aug
