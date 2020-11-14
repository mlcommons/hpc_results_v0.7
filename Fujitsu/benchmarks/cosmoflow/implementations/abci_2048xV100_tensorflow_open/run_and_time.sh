#!/bin/bash

module purge
module load gcc/7.4.0 python/3.7/3.7.6 cuda/10.1/10.1.243
module load cudnn/7.6/7.6.5 nccl/2.6/2.6.4-1 openmpi/2.1.6

. venv/bin/activate

cd ../implementation_abci_old

DataDir="/groups1/gca50115/tabuchi/cosmoflow/partial_small_dataset/cosmoUniverse_2019_05_4parE_tf_64"
StageDir="/dev/shm"

mpirun -np 2048 --mca btl openib -mca pml ob1 -mca mpi_warn_on_fork 0 \
python3 train.py configs/cosmo_open_1024node.yaml --data-dir $DataDir --stage-dir $StageDir --rank-gpu -d
