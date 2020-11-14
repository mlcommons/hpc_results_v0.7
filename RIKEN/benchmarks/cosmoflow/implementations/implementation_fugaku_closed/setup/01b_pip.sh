#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=dvsys-huge"
#PJM -L elapse=02:30:00
#PJM -L "node=1"
#PJM --mpi "proc=1"
#PJM -j
#PJM -S

set -ex

module load lang/tcsds-1.2.27b

. ../setenv
PREFIX="${COSMOFLOW_BASE}/opt"

pip3 install --upgrade pip

free -h

pip3 install --upgrade pip
pip3 install six wheel setuptools mock 'future>=0.17.1'
pip3 install cython pyyaml pandas==1.0.5 typeguard

pip install git+https://github.com/mlperf/logging.git@0.7.1
#MPICC=mpifcc pip install mpi4py
pip install /home/g9300001/u93182/up/numpy-1.18.4-cp38-cp38-linux_aarch64.whl

echo "#end"
