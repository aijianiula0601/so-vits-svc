#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../


cpt_file="/workspace/hjh/pycharm_projects/vcc/so-vits-svc/logs/44k/G_10000.pth"
python -u inference_main.py -m ${cpt_file} -c "configs/config.json" -n "test.wav" -t 0 -s "p334"



# 例
#python inference_main.py -m "logs/44k/G_10000.pth" -c "configs/config.json" -n "test.wav" -t 0 -s "nen"