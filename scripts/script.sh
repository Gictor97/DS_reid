#!/usr/bin/env bash
source=$1
target=$2

arch=$3
if [ $# -ne 3 ]
 then
   echo "Arguments error: <source> <target> <arch>"
   exit
fi # market1501 dukemtmc resnet50_idm

#CUDA_VISIBLE_DEVICES="1,2" python3 ./example/train_idm.py -ds ${source} -dt ${target} --mu2 0  --a1 0.7 \
#-a ${arch} -pool 'gpp'  --logs-dir ./logs/div/${source}-TO-${target}_${arch}_gpp_m3
CUDA_VISIBLE_DEVICES="2,1" python3 ./example/train_idm.py -ds ${source} -dt ${target} --mu2 0  --a1 0.1 \
-a ${arch}  --logs-dir ./logs/div/${source}-TO-${target}_${arch}_m3
CUDA_VISIBLE_DEVICES="2,1" python3 ./example/train_idm.py -ds ${source} -dt ${target} --mu2 0  --a1 0.1  \
-a ${arch} --pool 'gcp' --logs-dir ./logs/div/${source}-TO-${target}_${arch}_gcp_m3
#CUDA_VISIBLE_DEVICES="1,2" python3 ./example/train_idm.py -ds ${source} -dt ${target} --mu2 0  --a1 0.7 \
#-a ${arch} -pool 'gpp'  --logs-dir ./logs/div/${source}-TO-${target}_${arch}_gpp_m3128