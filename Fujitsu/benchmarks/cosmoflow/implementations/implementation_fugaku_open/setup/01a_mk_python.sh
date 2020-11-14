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

#export CC="fcc -Nclang -Kfast -Knolargepage"
#export CXX="FCC -Nclang -Kfast -Knolargepage"
export OPT=-O3
export ac_cv_opt_olimit_ok=no
export ac_cv_olimit_ok=no
export ac_cv_cflags_warn_all=''

PYTHON_NAME="python-3.8.5"

cd ${COSMOFLOW_BASE}/setup/Python-3.8.5

./configure --enable-shared --disable-ipv6 --target=aarch64 --build=aarch64 --prefix=$PREFIX/$PYTHON_NAME
make -j16
mv python python_org
${CXX} --linkfortran -SSL2 -Kopenmp -Nlibomp -o python Programs/python.o -L. -lpython3.8 -ldl  -lutil   -lm
make install

pip3 install --upgrade pip

python3 -V
ls -la $PREFIX/$PYTHON_NAME/bin

echo "#end"
