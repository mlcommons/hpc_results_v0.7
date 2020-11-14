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

#git clone https://github.com/mpi4py/mpi4py.git
##riken patch and change for 1.2.27b
#git clone git@kaiseki-juku.parc.flab.fujitsu.co.jp:postk_dl/mpi4py.git mpi4pytest
#git checkout fjdev
cd mpi4py
python setup.py install

git status

pip list

echo "#end"
