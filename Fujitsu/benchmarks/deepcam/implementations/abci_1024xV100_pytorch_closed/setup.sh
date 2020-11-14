#!/bin/bash

conda create -n py37-pytorch python=3.7
conda activate py37-pytorch

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install h5py
conda install -c conda-forge basemap
conda install basemap-data-hires
conda install matplotlib=3.2
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git

#Apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

#Logger
git clone -b hpc-0.5.0 https://github.com/mlperf-hpc/logging.git mlperf-logging
pip install [--user] -e mlperf-logging
