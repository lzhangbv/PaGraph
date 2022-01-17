#!/bin/bash

server="${server:-0}"

# settings
gpu_num=2
gpu_ids=0,1
preprocess=1
remote_sample=1
one2all=0

# partitions
pa_server=0
pa_trainer=0

if [ "$server" = "1" ]; then
gpu_num=$gpu_num preprocess=$preprocess remote_sample=$remote_sample one2all=$one2all pa_server=$pa_server bash launch_server.sh
else
gpu_ids=$gpu_ids preprocess=$preprocess remote_sample=$remote_sample one2all=$one2all pa_trainer=$pa_trainer bash run_trainer.sh
fi
