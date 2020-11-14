#!/bin/bash

SRC_DIR=$1
DST_DIR=$2
BASENAME_FILE=$3
WORLD_SIZE=$4
NfilesPerTar=$5
Type=$6

Rank=${PMIX_RANK}

export PATH=/data/g9300001/MLPerf/kro/CosmoFlow/run/script30:$PATH

NLINES=`wc -l $BASENAME_FILE | cut -d " " -f 1`
LOCAL_NLINES=$(( $NLINES / ${WORLD_SIZE} ))
NumLoops=$(( ${LOCAL_NLINES} / ${NfilesPerTar} ))


cd $SRC_DIR

LoopCount=0
Offset=$(( ${Rank} * ${LOCAL_NLINES} + 1))
while [ ${LoopCount} -lt ${NumLoops} ]; do
  FileID=$(( ${Offset} / ${NfilesPerTar} ))

  EndLine=$(( ${Offset} + ${NfilesPerTar} - 1))
  TargetFiles=`sed -n ${Offset},${EndLine}p ${BASENAME_FILE}`
  echo "Rank ${PMIX_RANK} Start: ${Offset} End: ${EndLine} FileID: ${FileID}"
  tar --use-compress-program=pixz -cf ${DST_DIR}/cosmo_${Type}_${FileID}.tar.xz ${TargetFiles}
  #tar -I pigz -cf ${DST_DIR}/cosmo_${Type}_${FileID}.tar.gz ${TargetFiles}
  #tar --use-compress-program="/groups1/gac50489/local/bin/pixz" -cf ${DST_DIR}/${basename}.tar.xz ./${basename}_*.tfrecord

  LoopCount=$(( ${LoopCount} + 1 ))
  Offset=$(( ${Offset} + ${NfilesPerTar} ))

done

