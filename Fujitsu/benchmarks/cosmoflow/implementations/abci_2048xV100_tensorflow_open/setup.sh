#!/bin/bash

module purge
module load gcc/7.4.0 python/3.7/3.7.6 cuda/10.1/10.1.243
module load cudnn/7.6/7.6.5 nccl/2.6/2.6.4-1 openmpi/2.1.6

python3 -m venv venv
. venv/bin/activate

pip install --upgrade pip
pip3 install setuptools==41.0.0
pip3 install tensorflow-gpu==2.2.0 pyyaml pandas
HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_MPI=1 HOROVOD_NCCL_HOME=$NCCL_HOME HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod==0.19.5

#Logger
git clone -b hpc-0.5.0 https://github.com/mlperf-hpc/logging.git mlperf-logging
pip install [--user] -e mlperf-logging
