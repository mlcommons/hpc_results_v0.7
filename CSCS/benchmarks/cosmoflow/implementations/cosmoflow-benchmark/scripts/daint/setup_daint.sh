# Source this script to setup the runtime environment on Daint

module load daint-gpu
module load Horovod/0.19.1-CrayGNU-20.08-tf-2.2.0

# Environment variables needed by the NCCL backend
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1

export MPICH_RDMA_ENABLED_CUDA=1

export PYTHONPATH=/scratch/snx3000/lukasd/mlperf/logging/:${PYTHONPATH}
