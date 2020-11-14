#!/bin/bash

SRC_DIR=$1
DST_DIR=$2
BASENAME_FILE=$3

LOCAL_BASENAME_FILE=$SGE_LOCALDIR/basenames_$OMPI_COMM_WORLD_RANK

NLINES=`wc -l $BASENAME_FILE | cut -d " " -f 1`
LOCAL_NLINES=$(( $NLINES / $OMPI_COMM_WORLD_SIZE ))
BEGIN_LINE=$(( $LOCAL_NLINES * $OMPI_COMM_WORLD_RANK + 1 ))
END_LINE=$(( $LOCAL_NLINES * ($OMPI_COMM_WORLD_RANK + 1) ))
#echo nlines $NLINES
#echo $LOCAL_NLINES, $BEGIN_LINE, $END_LINE


sed -n ${BEGIN_LINE},${END_LINE}p $BASENAME_FILE > $LOCAL_BASENAME_FILE
#echo sed -n ${BEGIN_LINE},${END_LINE}p $BASENAME_FILE to $LOCAL_BASENAME_FILE
#echo rank $OMPI_COMM_WORLD_RANK, `wc -l $LOCAL_BASENAME_FILE`
#echo rank $OMPI_COMM_WORLD_RANK, `head -n 1 $LOCAL_BASENAME_FILE`

Exec(){
    echo "$@"
#    "$@"
}

cd $SRC_DIR

while read basename
do
    echo $basename
    #tar -I pigz -cf ${DST_DIR}/${basename}.tar.gz ./${basename}_*.tfrecord
    #tar --use-compress-program=lz4 -cf ${DST_DIR}/${basename}.tar.lz4 ./${basename}_*.tfrecord
    tar --use-compress-program="/groups1/gac50489/local/bin/pixz" -cf ${DST_DIR}/${basename}.tar.xz ./${basename}_*.tfrecord
done < $LOCAL_BASENAME_FILE


