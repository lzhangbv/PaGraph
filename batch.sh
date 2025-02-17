#!/bin/bash

server="${server:-0}"

# settings
gpu_num=1
gpu_ids=0
preprocess=1
remote_sample=1

# partitions
pa_server=1
pa_trainer=1

# used for global shuffle
one2all=0

# training params
epochs=10

if [ "$one2all" = "1" ]; then
    remote_sample=1
    pa_server=0
    pa_trainer=0
fi


if [ "$server" = "1" ]; then
gpu_num=$gpu_num preprocess=$preprocess remote_sample=$remote_sample one2all=$one2all pa_server=$pa_server epochs=$epochs bash launch_server.sh
else
gpu_ids=$gpu_ids preprocess=$preprocess remote_sample=$remote_sample one2all=$one2all pa_trainer=$pa_trainer epochs=$epochs bash run_trainer.sh
fi
