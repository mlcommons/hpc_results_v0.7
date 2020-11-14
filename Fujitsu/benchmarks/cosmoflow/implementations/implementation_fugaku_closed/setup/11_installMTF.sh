#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=dvsys-huge"
#PJM -L elapse=00:30:00
#PJM -L "node=1"
#PJM --mpi "proc=1"
#PJM -j
#PJM -S

set -ex

module load lang/tcsds-1.2.27b
env

. ../setenv

cd ${COSMOFLOW_BASE}/mesh
git status

python setup.py install

pip list

echo "#end"
