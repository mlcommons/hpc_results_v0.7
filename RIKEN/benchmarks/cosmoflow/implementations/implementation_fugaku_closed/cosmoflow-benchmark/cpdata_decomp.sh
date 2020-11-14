#!/bin/bash

set -ex

if [ $# -ne 5 ]; then
  echo "cpdata argument error" 
  exit -1
fi

SourceDir=${1}
LocalDir=${2}
Compress=${3}
_SIZE=${4}
GroupSize=${5} # the number of nodes in a staging group

#_SIZE=${OMPI_UNIVERSE_SIZE}
_RANK=${PMIX_RANK}
_LOCAL_RANK=0
_LOCAL_SIZE=1

umask 007

err_data(){
echo `hostname` `ls -l ${LocalDir}/$1`
}

CopyOneFile(){
#    echo "file $1 copy now"
  cp ${SourceDir}/$1 ${LocalDir}/$1 || err_data
  chmod 666 ${LocalDir}/$1
}

Exec(){
    echo "$@"
    "$@"
}

case $Compress in
    gz | gzip) CompressProgram="pigz"
               CompressExt="gz" ;;
         lz4 ) CompressProgram="lz4"
               CompressExt="lz4" ;;
          xz ) CompressProgram="xz"
               CompressExt="xz" ;;
            *) echo "Unsupported compression type $Compress"
               exit 0;;
esac

NodeNum=$(( $_RANK / $_LOCAL_SIZE ))
if [ $(( $_SIZE % $_LOCAL_SIZE )) -ne 0 ] ; then
    echo "# processes is not a multiple of # nodes"
    exit 1
fi

if [ $_LOCAL_RANK -ne 0 ] ; then
    exit 0
fi

NumNodes=$(( $_SIZE / $_LOCAL_SIZE ))

if [ $(( $NumNodes % $GroupSize )) -ne 0 ] ; then
    echo "# nodes is not a multiple of # groups"
    exit 1
fi
NumGroups=$(( $NumNodes / $GroupSize ))
GroupNum=$(( $NodeNum / $GroupSize ))


for type in train validation ; do
    mkdir -p ${LocalDir}/${type}
    rm -f ${LocalDir}/${type}/*

    BASENAME_FILE=${SourceDir}/${type}_basenames
    LOCAL_BASENAME_FILE=${LocalDir}/${type}_basenames_$NodeNum

    #if [ $type = "train" ] ; then
    NumDist=$NumGroups
    DistIdx=$GroupNum
    #else
	#NumDist=$NumNodes
	#DistIdx=$NodeNum
    #fi

    NLINES=`wc -l $BASENAME_FILE | cut -d " " -f 1`
    LOCAL_NLINES=$(( $NLINES / $NumDist ))
    BEGIN_LINE=$(( $LOCAL_NLINES * $DistIdx + 1 ))
    END_LINE=$(( $LOCAL_NLINES * ($DistIdx + 1) ))

    sed -n ${BEGIN_LINE},${END_LINE}p $BASENAME_FILE > $LOCAL_BASENAME_FILE

    while read basename
    do
	echo ${type}/$basename
	tar --use-compress-program=${CompressProgram} -xf ${SourceDir}/${type}/${basename}.tar.${CompressExt} -C ${LocalDir}/${type}
    done < $LOCAL_BASENAME_FILE

done

echo Node $NodeNum has train:`ls -1 ${LocalDir}/train | wc -l` val:`ls -1 ${LocalDir}/validation | wc -l`

exit 0
