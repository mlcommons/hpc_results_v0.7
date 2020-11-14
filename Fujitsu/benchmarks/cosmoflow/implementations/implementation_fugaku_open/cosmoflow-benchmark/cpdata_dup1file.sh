#!/bin/bash

#sleep $OMPI_COMM_WORLD_RANK

#

if [ $# -ne 3 ]; then
  echo "cpdata error" 
  exit -1
fi

SourceDir=${1}
LocalDir=${2}
NumDup=${3}

umask 007

err_data(){
echo `hostname` `ls -l ${LocalDir}/$1`
}

CopyOneFile(){
#    echo "file $1 copy now"
  cp ${SourceDir}/$1 ${LocalDir}/$1 || err_data

}

Exec(){
    echo "$@"
    "$@"
}

mkdir ${LocalDir}/train
ln -s ${LocalDir}/train ${LocalDir}/validation
cp ${SourceDir}/train/univ_ics_2019-03_a11088_000.tfrecord ${LocalDir}/univ_ics_2019-03_a11088_000.tfrecord
SRC=${LocalDir}/univ_ics_2019-03_a11088_000.tfrecord
for i in `seq $NumDup`; do
    i=$(printf "%04d\n" "$i")
    DST=$(basename $SRC)
    DST=${LocalDir}/train/${DST%.*}_${i}.tfrecord
    cp $SRC $DST || err_data
done

exit 0
