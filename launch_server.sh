#!/bin/bash

gpu_num="${gpu_num:-1}"
preprocess="${preprocess:-1}"
remote_sample="${remote_sample:-1}"
one2all="${one2all:-0}"
pa_server="${pa_server:-1}"

params="--dataset /localdata/reddit --num-workers $gpu_num"

if [ "$preprocess" = "1" ]; then
params="$params --preprocess"
fi

if [ "$remote_sample" = "1" ]; then
params="$params --sample"
fi

if [ "$one2all" = "1" ]; then
params="$params --one2all"
fi

PY=/home/esetstore/dgl0.4/bin/python

# Store Server
#export OMP_NUM_THREADS=16

if [ "$pa_server" = "1" ]; then
    # PaGraph Store Server
    echo "python server/pa_server.py $params"
    $PY server/pa_server.py $params
else
    # DGL+Cache Store Server
    echo "python server/cache_server.py $params"
    $PY server/cache_server.py $params
fi
