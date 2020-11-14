#!/bin/bash

#$ -l rt_F=32
#$ -l h_rt=1:30:0
#$ -j y
#$ -cwd

deactivate 2>/dev/null
source /etc/profile.d/modules.sh

module load openmpi/2.1.6

SRC_DIR=/groups1/gac50489/datasets/cosmoflow_full/cosmoUniverse_2019_05_4parE_tf
DST_DIR=/bb/gac50489/datasets/cosmoflow_full/cosmoUniverse_2019_05_4parE_tf/tar_xz_64
CUR_DIR=$(pwd)

echo $CUR_DIR

for type in train validation ; do
    echo $type
    mkdir -p $DST_DIR/$type
    #echo nhosts $NHOSTS
    #cat $SGE_JOB_HOSTLIST
    sed -e 's/$/ slots=4/' $SGE_JOB_HOSTLIST > "$SGE_LOCALDIR/hostfile"
    echo "`date +%s.%N` #packing ${SRC_DIR}/$type to ${DST_DIR}/$type start at `date`"
    mpirun -n $(($NHOSTS * 4)) --hostfile $SGE_LOCALDIR/hostfile pack_core.sh ${SRC_DIR}/$type ${DST_DIR}/$type $CUR_DIR/full_${type}_basenames
    echo "`date +%s.%N` #packing $type data end at `date`"
done
