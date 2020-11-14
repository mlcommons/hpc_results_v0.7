#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=dvsys-huge"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM --mpi "proc=1"
#PJM -j
#PJM -S

set -ex

module load lang/tcsds-1.2.27b

. ../setenv

BAZEL_DIR=${COSMOFLOW_BASE}/setup/bazel_2.0.0
export HDF5_DIR=${COSMOFLOW_BASE}/opt/hdf5-1.10.7
export LD_LIBRARY_PATH=${HDF5_DIR}/lib:$LD_LIBRARY_PATH
export PATH=${BAZEL_DIR}:${HDF5_DIR}/bin:$PATH
export INCLUDE=${HDF5_DIR}/include:$INCLUDE
export CPATH=${HDF5_DIR}/include:$CPATH

export CC="fcc -Nclang -Kfast -Knolargepage"
export CXX="FCC -Nclang -Kfast -Knolargepage"

# for scipy install
export LAPACK=/opt/FJSVxtclanga/tcsds-1.2.27b/lib64/libfjlapack.so
export BLAS=${LAPACK}

pip list

#echo "# scipy"
#pip install scipy==1.4.1

echo "# grpc build"
pushd ${COSMOFLOW_BASE}/setup/grpc
python3 setup.py install

echo "# libhdf5 build"
INSTALL_DIR=${COSMOFLOW_BASE}/setup/cmake
pushd ${COSMOFLOW_BASE}/setup/cmake-3.18.2
mkdir -p build; cd build
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} ..
make -j 20; make install
export PATH=${INSTALL_DIR}/bin:${PATH}

cd ${COSMOFLOW_BASE}/setup/hdf5-1.10.7
INSTALL_DIR=${COSMOFLOW_BASE}/opt/hdf5-1.10.7
mkdir -p build; cd build
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DHDF5_BUILD_TOOLS:BOOL=ON ..
make -j 20 && make install
popd

which cmake
ldd ${INSTALL_DIR}/lib/libhdf5.so

pip install h5py==2.10.0
pip list

echo "#end"
