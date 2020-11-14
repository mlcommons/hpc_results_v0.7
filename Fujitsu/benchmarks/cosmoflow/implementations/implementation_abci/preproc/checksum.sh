#!/bin/bash

#$ -l rt_F=8
#$ -l h_rt=1:00:0
#$ -j y
#$ -cwd

deactivate 2>/dev/null
source /etc/profile.d/modules.sh

module load openmpi/2.1.6

# setting
DATA_DIR=/groups1/gac50489/datasets/cosmoflow_full/cosmoUniverse_2019_05_4parE_tf
KIND=validation

CUR_DIR=$(pwd)

sed -e 's/$/ slots=16/' $SGE_JOB_HOSTLIST > "$SGE_LOCALDIR/hostfile"
mpirun -n $(($NHOSTS * 16)) --hostfile $SGE_LOCALDIR/hostfile checksum_core.sh ${DATA_DIR}/${KIND} ${CUR_DIR}/full_${KIND}_files


