#!/usr/bin/env bash
#T4 20220802
source=$1
target=$2
arch=$3
if [ $# -ne 3 ]
 then
   echo "Arguments error: <source> <target> <arch>"
   exit 1
fi # market1501 msmt17 resnet50_idm
CUDA_VISIBLE_DEVICES='1' python3 ./example/train_idm.py -ds ${source} -dt ${target} -a ${arch} --pool 'gcp' \
--logs-dir ./example/logs/${source}-TO-${target}_${arch}_${pool}
CUDA_VISIBLE_DEVICES='1' python3 ./example/train_idm.py -ds ${source} -dt ${target} -a ${arch} --pool 'gpp' \
--logs-dir ./example/logs/${source}-TO-${target}_${arch}_${pool}
CUDA_VISIBLE_DEVICES='1' python3 ./example/train_idm.py -ds ${source} -dt ${target} -a ${arch}  \
--logs-dir ./example/logs/${source}-TO-${target}_${arch}
CUDA_VISIBLE_DEVICES='1' python3 ./example/train_baseline.py -ds ${source} -dt ${target} -a resnet50 \
--logs-dir ./example/logs/${source}-TO-${target}_baseline
