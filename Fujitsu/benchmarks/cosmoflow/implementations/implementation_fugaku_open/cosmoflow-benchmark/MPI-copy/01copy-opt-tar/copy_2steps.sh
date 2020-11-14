#!/bin/bash

Nnodes=$1 && shift
Nroots=$1 && shift
TargetFile=$1 && shift
SrcDir=$1 && shift
DstDir=$1 && shift
HostFile=$1 && shift
RootsFile=$1 

ExeProg="mpi_bcast"

module load lang/tcsds-1.2.27b

Host=`hostname -i | awk '{print $NF}'`
grep $Host "${RootsFile}" > /dev/null 2>&1

if [ $? = 1 ]; then
  Rank=`grep ${Host} -n ${HostFile} | cut -d ":" -f 1`
  NodesPerGroup=$(( ${Nnodes} / ${Nroots} ))
  Tmp=$(( ${Rank} - 1 ))

  Tmp=$(( ${Tmp} / ${NodesPerGroup} ))
  Tmp=$(( ${Tmp} + 1 ))
  if [ ${Tmp} -gt ${Nroots} ]; then
    Tmp=$(( ${Rank} % ${Nroots} ))
    Tmp=$(( ${Tmp} + 1 ))
  fi

  export GroupID=${Tmp}
  export Roots=0
  export TargetFilePath=${DstDir}/${TargetFile}

  mkdir -p ${DstDir}
else
  SrcFile=${SrcDir}/${TargetFile}
  export GroupID=`grep ${Host} -n ${RootsFile} | cut -d ":" -f 1`
  export Roots=1
  export TargetFilePath=${DstDir}/${TargetFile}

  mkdir -p ${DstDir}
  ### Step1
  cp ${SrcFile} ${DstDir}
fi

### Step2
./${ExeProg}

