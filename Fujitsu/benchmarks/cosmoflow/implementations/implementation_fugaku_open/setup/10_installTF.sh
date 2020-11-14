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

WhlDir=/home/g9300001/u93182/up
# for scipy install
export LAPACK=/opt/FJSVxtclanga/tcsds-1.2.27b/lib64/libfjlapack.so
export BLAS=${LAPACK}

pip3 install --upgrade pip
pip3 install six wheel setuptools mock 'future>=0.17.1'
pip3 install cython pyyaml pandas==1.0.5

pip3 install keras_applications --no-deps    #need h5py
pip3 install keras_preprocessing --no-deps

pip3 install ${WhlDir}/tensorflow-2.2.0-cp38-cp38-linux_aarch64.whl

echo "#end"
