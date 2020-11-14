#!/bin/bash

set -e

if [ $# -ne 5 ]; then
  echo "cpdata argument error" 
  exit -1
fi

# path to pixz
export PATH=/data/g9300001/MLPerf/kro/CosmoFlow/run/script30:$PATH
TotalNumTarFiles=8192  # Total number of tar files

SourceDir=${1}
LocalDir=${2}
Compress=${3}
_SIZE=${4}
GroupSize=${5}
_RANK=${PMIX_RANK}

SourceDir=/vol0001/input_MLPerf/${SourceDir}
umask 007

case $Compress in
    gz | gzip) CompressProgram="pigz"
               CompressExt="gz" ;;
         lz4 ) CompressProgram="lz4"
               CompressExt="lz4" ;;
          xz ) CompressProgram="pixz"
               CompressExt="xz" ;;
            *) echo "Unsupported compression type $Compress"
               exit 0;;
esac

NodeNum=$_RANK
NumNodes=$_SIZE

if [ $(( $NumNodes % $GroupSize )) -ne 0 ] ; then
    echo "# nodes is not a multiple of # groups"
    exit 1
fi
NumGroups=$(( $NumNodes / $GroupSize ))
GroupNum=$(( $NodeNum / $GroupSize ))

for type in train validation ; do
    mkdir -p ${LocalDir}/${type}
    rm -f ${LocalDir}/${type}/*


    NumTarFiles=$(( ${TotalNumTarFiles} / ${NumGroups} ))
    Offset=$(( ${GroupNum} * ${NumTarFiles} ))
    EndIdx=$(( ${Offset} + ${NumTarFiles} - 1 ))
    for Idx in `seq ${Offset} ${EndIdx}`
    do
        DirIdx=$(( ${Idx} / 1000 ))
        #time -p tar  --use-compress-program=${CompressProgram} -p 46 -xf ${SourceDir}/${type}/${DirIdx}/cosmo_${type}_${Idx}.tar.${CompressExt} -C ${LocalDir}/${type}
        #time -p tar -I "${CompressProgram} -p 46" -xf ${SourceDir}/${type}/${DirIdx}/cosmo_${type}_${Idx}.tar.${CompressExt} -C ${LocalDir}/${type}
        tar -I "${CompressProgram} -p 46" -xf ${SourceDir}/${type}/${DirIdx}/cosmo_${type}_${Idx}.tar.${CompressExt} -C ${LocalDir}/${type}
    done

done

echo Node $NodeNum has train:`ls -1 ${LocalDir}/train | wc -l` val:`ls -1 ${LocalDir}/validation | wc -l`

exit 0
