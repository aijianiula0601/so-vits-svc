#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../

sr2=44100

base_dir="/mnt/cephfs/hjh/train_record/vc/so-vits-svc/vctk/test_data"
in_dir="${base_dir}/debug_data"
out_dir2="${base_dir}/resample_to_44100"


#-------------------
#flac to wav
#-------------------
#for file in ${in_dir}/*/*.flac
#do
#  new_file=`echo ${file}|sed "s/.flac/.wav/"`
#  echo ${new_file}
#  ffmpeg -i ${file} ${new_file}
#  rm -rf ${file}
#done


#-------------------
# resample to 44100
#-------------------

python -u resample.py --sr2 ${sr2} --in_dir ${in_dir} --out_dir2 ${out_dir2}
