#!/usr/bin/env bash
source=$1
target=$2
arch=$3
if [ $# -ne 3 ]
 then
   echo "Arguments error: <source> <target> <arch>"
   exit 1
fi # market1501,dukemtmc,msmt17,resnet50_idm
CUDA_VISIBLE_DEVICES='0' python3 ./example/train_idm.py -ds ${source} -dt ${target} -a ${arch} --pool 'gcp' \
--logs-dir ./example/logs/market-TO-${target}_${arch}_${pool}
CUDA_VISIBLE_DEVICES='0' python3 ./example/train_idm.py -ds ${source} -dt ${target} -a %{arch} \
--logs-dir ./example/lo1gs/market-TO-${target}_idm