#!/bin/bash

cd ../implementation_fugaku_closed/setup

# build Python for aarch64 and install oneAPI Deep Neural Network Library (oneDNN) before the following steps

# install python libraries
pjsub 01b_pip.sh

# build and install grpc and libhdf5+h5py
tar xf /home/g9300001/u93182/up/grpc.tgz
tar xf /home/g9300001/u93182/up/hdf5-1.10.7.tar.gz
tar xf /home/g9300001/u93182/up/cmake-3.18.2.tar.gz
pjsub 03_lib.sh

# build tensorflow
pjsub 04_buildTF.sh

# install mesh-tensorflow module
11_installMTF.sh

# install MPI4py
13_installMPI4py.sh
