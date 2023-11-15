#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../

sr2=$1
in_dir=$2
out_dir2=$3

python -u resample.py --sr2 ${sr2} --in_dir ${in_dir} --out_dir2 ${out_dir2}
