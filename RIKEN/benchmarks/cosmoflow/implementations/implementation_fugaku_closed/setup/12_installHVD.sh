#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=dvsys-huge"
#PJM -L elapse=00:30:00
#PJM -L "node=1"
#PJM --mpi "proc=1"
#PJM -j
#PJM -S

set -ex

module load lang/tcsds-1.2.27b

. ../setenv
PREFIX="${COSMOFLOW_BASE}/opt"

#pip uninstall -y horovod
export HOROVOD_MPICXX_SHOW="mpiFCC -show"
export HOROVOD_WITHOUT_MXNET=1
export HOROVOD_WITHOUT_GLOO=1
export HOROVOD_WITH_MPI=1
export HOROVOD_WITHOUT_PYTORCH=1
export HOROVOD_WITH_TENSORFLOW=1
pip install horovod==0.19.5 --no-cache-dir

horovodrun -cb

echo "#end"
