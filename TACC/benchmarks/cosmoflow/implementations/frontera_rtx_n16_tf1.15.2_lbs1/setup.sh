# Install Miniconda if not available on the target system 

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 755 Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh                 # You need to specify path (assumed as <Miniconda-Conda-DIR>) to install Miniconda
source <Miniconda-Conda-DIR>/miniconda/bin/activate # Activate Miniconda


# Setup steps for TACC FRONTERA-RTX nodes
# Login to a GPU node as some of the libraries are not available on login nodes

idev -N 1 -p rtx-dev
module load gcc/9.1.0 cuda/10.0 cudnn/7.6.2  cmake nccl/2.4.7 impi

conda create -y -n mlperf_cosmo_scratch3 python=3.7
conda list env

conda activate mlperf_cosmo_scratch3
conda env list

pip install  --upgrade pip
pip install  pandas
pip install  --no-cache-dir  tensorflow-gpu==1.15.2
pip install  --no-cache-dir  keras==2.2.4
HOROVOD_CUDA_HOME=$TACC_CUDA_DIR HOROVOD_NCCL_HOME=$TACC_NCCL_DIR CC=gcc HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_TENSORFLOW=1 pip3 install --user horovod==0.19.2 --no-cache-dir --force-reinstall

cd ..
git clone -b hpc-0.5.0 https://github.com/mlperf-hpc/logging.git mlperf-logging
pip install -e mlperf-logging


