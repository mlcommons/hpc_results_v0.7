#!/bin/bash

conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi
conda install -y -c magma-cuda102

conda remove pytorch

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CUDNN_INCLUDE_DIR=/apps/cudnn/7.6.5/cuda10.2/include
export CUDNN_LIBRARY=/apps/cudnn/7.6.5/cuda10.2/lib64

# fj_pytorch
pushd ../implementation_abci_fj

tar zxf fj_pytorch_1.6.0.tar.gz
pushd fj_pytorch_1.6.0
python setup.py install
conda install -y cudatoolkit=10.2 -c pytorch
popd 

git clone --recursive https://github.com/pytorch/vision.git
pushd vision
git checkout tags/v0.7.0
python setup.py install
popd 

popd 

conda install h5py
conda install -c conda-forge basemap
conda install basemap-data-hires
conda install matplotlib=3.2
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git

#Apex
pushd ../implementation_abci_fj
git clone https://github.com/NVIDIA/apex.git
cp fused_lamb.py apex/apex/optimizers/fused_lamb.py
pushd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
popd 
popd 

#Logger
git clone -b hpc-0.5.0 https://github.com/mlperf-hpc/logging.git mlperf-logging
pip install [--user] -e mlperf-logging
