#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=eap-small"
#PJM -L elapse=02:00:00
#PJM -L "node=1"
#PJM --mpi "proc=1"
#PJM -j
#PJM -S

set -ex

module load lang/tcsds-1.2.27b

. ../setenv

BAZEL_DIR=${COSMOFLOW_BASE}/setup/bazel_2.0.0
export PATH=${BAZEL_DIR}:$PATH

export CC="fcc -Nclang -Kfast -Knolargepage"
export CXX="FCC -Nclang -Kfast -Knolargepage"

# for scipy install
#export LAPACK=/opt/FJSVxtclanga/tcsds-1.2.27b/lib64/libfjlapack.so
#export BLAS=${LAPACK}

echo "# Tensorflow build"
pushd ${COSMOFLOW_BASE}/setup/tensorflow
/bin/yes '' | ./configure

./fcc_build_script/FX700_02_fccTF-dnnl_aarch64_build_v2.1.0.sh

bazel-bin/tensorflow/tools/pip_package/build_pip_package .
pip install --upgrade *.whl

pip list

popd
echo "#end"
