#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../

#CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -u train_my.py -c configs/config.json -m 44k
