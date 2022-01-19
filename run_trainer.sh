#!/bin/bash

gpu_ids="${gpu_ids:-0}"
preprocess="${preprocess:-1}"
remote_sample="${remote_sample:-1}"
one2all="${one2all:-0}"
pa_trainer="${pa_trainer:-1}"

params="--dataset /localdata/reddit --feat-size 602 --gpu $gpu_ids"

if [ "$preprocess" = "1" ]; then
params="$params --preprocess"
fi

if [ "$remote_sample" = "1" ]; then
params="$params --remote-sample"
fi

if [ "$one2all" = "1" ]; then
params="$params --one2all"
fi

PY=/home/esetstore/dgl0.4/bin/python
#export OMP_NUM_THREADS=16

# benchmark
if [ "$pa_trainer" = "0" ]; then
    # (1) dgl gcn
    echo "python examples/profile/dgl_gcn.py $params"
    $PY examples/profile/dgl_gcn.py $params
    # (2) dgl + cache gcn
    #echo "python examples/profile/dgl_cache.py $params"
    #$PY examples/profile/dgl_cache.py $params
else
    # (3) pagraph gcn
    echo "python examples/profile/pa_gcn.py $params"
    $PY examples/profile/pa_gcn.py $params
fi
