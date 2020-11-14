#!/bin/bash -l
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH -J train-cosmoflow-daint
#SBATCH -o logs/%x-%j.out
# #SBATCH -d singleton

. scripts/daint/setup_sarus.sh

# Data staging skipped

# Run the training
set -x
srun -l -u sarus run --mpi \
    --mount=type=bind,source=$(pwd)/../data,destination=/root/mlperf/data \
    --mount=type=bind,source=$(pwd),destination=/root/mlperf/cosmoflow-benchmark \
    --workdir=/root/mlperf/cosmoflow-benchmark \
    load/library/cosmoflow_gpu_daint \
    python train.py --distributed --rank-gpu $@
