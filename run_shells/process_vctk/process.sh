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
#bash split_dataset.sh $out_dir2 $train_file $val_file


echo '--------------------------'
echo '第三步:'
echo '--------------------------'
#bash preprocess_hubert_f0.sh ${out_dir2}



echo '--------------------------'
echo '检查数据'
echo '--------------------------'
check_function(){
    reading_file=$1
    tmp_file=$2
    rm -rf $tmp_file

    for wav_f in `cat ${reading_file}`
    do
      f0_f=`echo ${wav_f}|sed 's/.wav/.wav.f0.npy/'`
      soft_f=`echo ${wav_f}|sed 's/.wav/.wav.soft.pt/'`
      vol_f=`echo ${wav_f}|sed 's/.wav/.wav.vol.npy/'`

      if [ -f "${wav_f}" -a -f "${f0_f}" -a -f "${soft_f}" -a -f "${vol_f}" ];then
        echo ${wav_f} >> ${tmp_file}
      fi
    done

    echo "save to ${tmp_file}"

}

#tmp_file="${base_dir}/val_check.txt"
#check_function $val_file ${tmp_file}

tmp_file="${base_dir}/train_check.txt"
check_function $train_file ${tmp_file}