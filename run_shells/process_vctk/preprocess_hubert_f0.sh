#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../


out_dir=$1

python preprocess_hubert_f0.py \
--in_dir ${out_dir} \
--f0_predictor dio \
--num_processes 40