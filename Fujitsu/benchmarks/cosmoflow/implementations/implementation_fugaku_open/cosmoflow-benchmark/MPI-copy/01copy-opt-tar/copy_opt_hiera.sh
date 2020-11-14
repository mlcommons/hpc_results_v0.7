#!/bin/bash 


SrcDir=$1 && shift
DstDir=$1 && shift
TargetFile=$1 && shift

HostFile=".hostfile"
HostStep1=".hoststep1"
ExeProg="copy_2steps.sh"

module load lang/tcsds-1.2.27b

export PLE_MPI_STD_EMPTYFILE="off"

# Get nodelist
Nnodes=`pjshowip | wc | awk '{print $1}'`
pjshowip > ${HostFile}


# Step1 
Tmp=${Nnodes}
CountBit=0
while [ ${Tmp} -gt 0 ]
do
    CountBit=$(( ${CountBit} + 1 ))
    Tmp=$(( ${Tmp} / 2 ))
done

HalfBit=$(( ${CountBit} / 2 ))
NnodesStep1=$((2 ** ${HalfBit}))
cat ${HostFile} | awk -v val=${NnodesStep1} '{ if (NR % val == 1) print; }' > ${HostStep1}

Nroots=`wc ${HostStep1} | awk '{print $1}'`
mpiexec -n ${Nnodes} -of copy_mpi_bcast.log \
        ./${ExeProg} ${Nnodes} ${Nroots} ${TargetFile} ${SrcDir} ${DstDir} ${HostFile} ${HostStep1}

