#!/bin/bash

set -e

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../

base_dir='/mnt/cephfs/hjh/train_record/vc/so-vits-svc/vctk/test_data'
in_dir="${base_dir}/debug_data"
meta_file="${base_dir}/debug_data.txt"
rm -rf ${meta_file}

echo '--------------------------'
echo '提取bert信息'
echo '--------------------------'
#python -u dataest_utils/preprocess_hubert.py \
#  --in_dir=$in_dir \
#  --num_processes=20

echo '--------------------------'
echo '检查'
echo '--------------------------'
check_function() {
  # shellcheck disable=SC2045
  for spk_name in $(ls $1); do
    spk_dir=$1"/"$spk_name
    if [ -d $spk_dir ]; then
      for wav_f in $(ls $spk_dir/*.wav); do
        soft_f="${wav_f}.soft.pt"
        if [ ! -f "${soft_f}" ]; then
          echo '---------------------error---------------------------'
          echo "wav file has not soft file: ${soft_f}"
          echo '-----------------------------------------------------'
          exit 1
        fi
        echo "${wav_f}|${soft_f}|${spk_name}" >> ${meta_file}
      done
    fi
  done
}

check_function ${in_dir}
echo "检查完毕，没问题！"
echo "mate file save to: ${meta_file}"
