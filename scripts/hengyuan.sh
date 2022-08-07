#!/usr/bin/env bash
## 20220805 T4
source=$1
target=$2
arch=$3
if [ $# -ne 3 ]
 then
   echo "Arguments error: <source> <target> <arch>"
   exit 1
fi  # dukemtmc market1501 resnet50_idm

CUDA_VISIBLE_DEVICES='1' python3 ./example/train_idm.py -ds ${source} -dt ${target} -a ${arch} \
--logs-dir ./example/logs/${source}-TO-${target}_idm
#CUDA_VISIBLE_DEVICES='1' python3 ./example/train_idm.py -ds ${source} -dt ${target} --gamma 0.0 -a ${arch} --pool 'gcp' \
#--logs-dir ./example/logs/${source}-TO-${target}_idm_gcp_0.0
## 20220805 T40