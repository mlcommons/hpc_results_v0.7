#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=dvsys-huge"
#PJM -L elapse=$TimeLimit
#PJM -L proc-core=48
#PJM -L "node=64"
#PJM --mpi "max-proc-per-node=1"
#PJM -j
#PJM -S

module load lang/tcsds-1.2.27b

## validation data: 65536
NumTrainData=65536
NumTarFiles=8192
NfilesPerTar=$(( ${NumTrainData} / ${NumTarFiles} ))

SRC_DIR=/data/g9300001/MLPerf/cosmoUniverse_2019_05_4parE_tf
DST_DIR=/data/g9300001/MLPerf/cosmoUniverse_${NumTarFiles}_tarfiles_xz
CUR_DIR=$(pwd)

echo $CUR_DIR

NHOSTS=`pjshowip | wc | awk '{print $1}'`

type="validation"
echo $type

mkdir -p $DST_DIR/$type
echo "`date +%s.%N` #packing ${SRC_DIR}/$type to ${DST_DIR}/$type start at `date`"

mpiexec -n $NHOSTS pack_core.sh \
    ${SRC_DIR}/$type ${DST_DIR}/$type \
    $CUR_DIR/${type}_list.txt ${NHOSTS} ${NfilesPerTar} ${type}

echo "`date +%s.%N` #packing $type data end at `date`"
