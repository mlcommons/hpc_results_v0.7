#!/bin/bash -l
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -t 00:40:00
#SBATCH -J train-cosmoflow-daint
#SBATCH -o logs/%x-%j.out
# #SBATCH -d singleton

. scripts/daint/setup_daint.sh

# Data staging skipped

# Run the training
set -x
srun -l -u python train.py --distributed --rank-gpu $@
