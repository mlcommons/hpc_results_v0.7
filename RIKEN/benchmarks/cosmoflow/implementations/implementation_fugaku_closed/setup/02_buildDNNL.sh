#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=dvsys-huge"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM --mpi "proc=1"
#PJM -j
#PJM -S

set -ex

. ../setenv

PREFIX="${COSMOFLOW_BASE}/opt"

echo "# clone oneDNN"
DNNL_SRC_DIR=${COSMOFLOW_BASE}/setup/oneDNN
#pushd ${COSMOFLOW_BASE}/setup
#git clone git@kaiseki-juku.parc.flab.fujitsu.co.jp:postk_dl/oneDNN.git
#cd oneDNN
#git checkout fccbuild
#git submodule update --init --recursive
#popd

echo "# XED build"
XED_BUILD_DIR=${PREFIX}/xed_aarch64
#export XED_ROOT_DIR=${XED_BUILD_DIR}/kits/xed
#export C_INCLUDE_PATH=${XED_ROOT_DIR}/include${C_INCLUDE_PATH}
#export LD_LIBRARY_PATH=${XED_ROOT_DIR}/lib:${LD_LIBRARY_PATH}

mkdir -p ${XED_BUILD_DIR}
pushd ${XED_BUILD_DIR}
${DNNL_SRC_DIR}/src/cpu/aarch64/xbyak_translator_aarch64/translator/third_party/xed/mfile.py --shared examples install
cd kits
ln -sf xed-install-base-* xed
popd

pushd ${COSMOFLOW_BASE}/setup/oneDNN
DNNL_BUILD_DIR=${PREFIX}/oneDNN-build
#mv ${DNNL_BUILD_DIR} ${DNNL_BUILD_DIR}_bak
#rm -rf ${DNNL_BUILD_DIR}
mkdir -p ${DNNL_BUILD_DIR}
ln -sf ${DNNL_BUILD_DIR} ./build

cd ${DNNL_BUILD_DIR}
cmake ${DNNL_SRC_DIR} -DDNNL_INDIRECT_JIT_AARCH64=ON -DWITH_BLAS=ssl2 2>&1 |tee -a cmake_fcc.sh.log

make -j 30
popd

mv -r ${DNNL_BUILD_DIR} ${COSMOFLOW_BASE}/opt/oneDNN
ldd ${DNNL_BUILD_DIR}/src/libdnnl.so

echo "#end"
