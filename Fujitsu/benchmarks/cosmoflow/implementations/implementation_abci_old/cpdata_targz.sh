#!/bin/bash

#sleep $OMPI_COMM_WORLD_RANK

#

if [ $# -ne 2 ]; then
  echo "cpdata error" 
  exit -1
fi

SourceDir=${1}
LocalDir=${2}

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

NODE_NUM=$(( $OMPI_COMM_WORLD_RANK / $OMPI_COMM_WORLD_LOCAL_SIZE ))
if [ $OMPI_COMM_WORLD_LOCAL_RANK -ne 0 ] ; then
    exit 0
fi

NNODES=$(( $OMPI_COMM_WORLD_SIZE / $OMPI_COMM_WORLD_LOCAL_SIZE ))


for type in train validation ; do
    mkdir -p ${LocalDir}/${type}

    BASENAME_FILE=${SourceDir}/${type}_basenames
    LOCAL_BASENAME_FILE=${LocalDir}/${type}_basenames_$NODE_NUM

    NLINES=`wc -l $BASENAME_FILE | cut -d " " -f 1`
    LOCAL_NLINES=$(( $NLINES / $NNODES ))
    BEGIN_LINE=$(( $LOCAL_NLINES * $NODE_NUM + 1 ))
    END_LINE=$(( $LOCAL_NLINES * ($NODE_NUM + 1) ))

    sed -n ${BEGIN_LINE},${END_LINE}p $BASENAME_FILE > $LOCAL_BASENAME_FILE

    while read basename
    do
	echo ${type}/$basename
	tar -I pigz -xf ${SourceDir}/${type}/${basename}.tar.gz -C ${LocalDir}/${type}
    done < $LOCAL_BASENAME_FILE

    #ls -l ${LocalDir}/${type}
done

exit 0
