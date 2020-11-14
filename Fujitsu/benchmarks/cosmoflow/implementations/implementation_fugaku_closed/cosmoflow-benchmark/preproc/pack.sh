#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=dvsys-huge"
#PJM -L elapse=10:00:00
#PJM -L proc-core=48
#PJM -L "node=64"
#PJM --mpi "max-proc-per-node=1"
#PJM -j
#PJM -S

#module load openmpi/2.1.6

SRC_DIR=/data/g9300001/MLPerf/cosmoUniverse_2019_05_4parE_tf
DST_DIR=/data/g9300001/MLPerf/cosmoUniverse_64each_gz
CUR_DIR=$(pwd)

echo $CUR_DIR
PROC=${PJM_MPI_PROC}

for type in train validation ; do
    echo $type
    mkdir -p $DST_DIR/$type
    #echo nhosts $NHOSTS
    #cat $SGE_JOB_HOSTLIST
    #sed -e 's/$/ slots=4/' $SGE_JOB_HOSTLIST > "$SGE_LOCALDIR/hostfile"
    echo "`date +%s.%N` #packing ${SRC_DIR}/$type to ${DST_DIR}/$type start at `date`"
    mpirun -n $PROC pack_core.sh ${SRC_DIR}/$type ${DST_DIR}/$type $CUR_DIR/full_${type}_basenames
    echo "`date +%s.%N` #packing $type data end at `date`"
done
