#!/bin/bash

if [ $# -ne 4 ]; then
  echo "cpdata argument error" 
  exit -1
fi

SourceDir=${1}
LocalDir=${2}
Compress=${3}
GroupSize=${4} # the number of nodes in a staging group

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

case $Compress in
    gz | gzip) CompressProgram="pigz"
               CompressExt="gz" ;;
         lz4 ) CompressProgram="lz4"
               CompressExt="lz4" ;;
          xz ) CompressProgram="/groups1/gac50489/local/bin/pixz"
               CompressExt="xz" ;;
            *) echo "Unsupported compression type $Compress"
               exit 0;;
esac

NodeNum=$(( $OMPI_COMM_WORLD_RANK / $OMPI_COMM_WORLD_LOCAL_SIZE ))
if [ $(( $OMPI_COMM_WORLD_SIZE % $OMPI_COMM_WORLD_LOCAL_SIZE )) -ne 0 ] ; then
    echo "# processes is not a multiple of # nodes"
    exit 1
fi

if [ $OMPI_COMM_WORLD_LOCAL_RANK -ne 0 ] ; then
    exit 0
fi

NumNodes=$(( $OMPI_COMM_WORLD_SIZE / $OMPI_COMM_WORLD_LOCAL_SIZE ))

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

    if [ $type = "train" ] ; then
	NumDist=$NumGroups
	DistIdx=$GroupNum
    else
	NumDist=$NumNodes
	DistIdx=$NodeNum
    fi

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
