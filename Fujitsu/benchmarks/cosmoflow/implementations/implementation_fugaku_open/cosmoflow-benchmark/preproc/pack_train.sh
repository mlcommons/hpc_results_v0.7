#!/bin/bash

#$ -l rt_F=32
#$ -l h_rt=1:30:0
#$ -j y
#$ -cwd

module load lang/tcsds-1.2.27b

## train data: 262144
NumTrainData=262144
NumTarFiles=8192
NfilesPerTar=$(( ${NumTrainData} / ${NumTarFiles} ))

SRC_DIR=/data/g9300001/MLPerf/cosmoUniverse_2019_05_4parE_tf
DST_DIR=/data/g9300001/MLPerf/cosmoUniverse_${NumTarFiles}_tarfiles_xz
CUR_DIR=$(pwd)

echo $CUR_DIR

NHOSTS=`pjshowip | wc | awk '{print $1}'`

type="train"
echo $type
mkdir -p $DST_DIR/$type

echo "`date +%s.%N` #packing ${SRC_DIR}/$type to ${DST_DIR}/$type start at `date`"

mpiexec -n $NHOSTS pack_core.sh \
    ${SRC_DIR}/$type ${DST_DIR}/$type \
    $CUR_DIR/${type}_list.txt ${NHOSTS} ${NfilesPerTar} ${type}

echo "`date +%s.%N` #packing $type data end at `date`"
