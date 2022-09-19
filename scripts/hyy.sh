#!/usr/bin/env bash
source=$1
target=$2
arch=$3

if [ $# -ne 3 ]
 then
   echo "Arguments error: <source> <target> <arch>"
   exit 1
fi # market1501 dukemtmc resnet50_idm 1.0
CUDA_VISIBLE_DEVICES='2,3' python3 ./example/train_idm.py -ds ${source} -dt ${target} -a ${arch}  \
	--logs-dir ./logs/mmd/${source}-TO-${target}_idm
CUDA_VISIBLE_DEVICES='2,3' python3 ./example/train_idm.py -ds ${source} -dt ${target} -a ${arch} -pool 'gcp' \
	--logs-dir ./logs/mmd/${source}-TO-${target}_idm_gcp
CUDA_VISIBLE_DEVICES='2,3' python3 ./example/train_idm.py -ds ${source} -dt ${target} -a ${arch} -pool 'gpp'\
	--logs-dir ./logs/mmd/${source}-TO-${target}_idm_gpp