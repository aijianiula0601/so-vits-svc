#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../


base_dir="/mnt/cephfs/hjh/train_record/vc/so-vits-svc/vctk/test_data"
out_dir="${base_dir}/resample_to_44100"

python preprocess_hubert_f0.py \
--in_dir ${out_dir} \
--f0_predictor dio