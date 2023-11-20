#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit


cd ../../

base_dir='/mnt/cephfs/hjh/train_record/vc/so-vits-svc/vctk/test_data'
in_dir="${base_dir}/debug_data"

echo '--------------------------'
echo '第一步:'
echo '--------------------------'
python -u dataest_utils/preprocess_hubert.py \
--in_dir=$in_dir \
--num_processes=8


