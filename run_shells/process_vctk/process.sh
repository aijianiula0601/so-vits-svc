#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit




base_dir="/mnt/cephfs/hjh/train_record/vc/so-vits-svc/vctk"
in_dir="${base_dir}/wav48_silence_trimmed"
out_dir2="${base_dir}/resample_to_44100"

echo '--------------------------'
echo '第一步:'
echo '--------------------------'
sr2=44100
#bash resample.sh $sr2 $in_dir $out_dir2


echo '--------------------------'
echo '第二步:'
echo '--------------------------'
train_file="${base_dir}/train.txt"
val_file="${base_dir}/val.txt"
bash split_dataset.sh $out_dir2 $train_file $val_file


echo '--------------------------'
echo '第三步:'
echo '--------------------------'
#bash preprocess_hubert_f0.sh ${out_dir2}