# Source this script to setup the container runtime environment on Daint

module load daint-gpu
module load sarus

export MPICH_RDMA_ENABLED_CUDA=1
