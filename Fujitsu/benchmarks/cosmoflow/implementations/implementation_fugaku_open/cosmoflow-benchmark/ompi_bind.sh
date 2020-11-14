#!/bin/bash

set -e

date

SIZE=${NumNodes}
RANK=${PMIX_RANK}
echo `date` "#start hostname: " `hostname` "JOBID: "${PJM_SUBJOBID}
#df -h /tmp /var/crash /local
rm -rf /tmp/${USER} >/dev/null 2>&1 :

### Copy opt.tgz file and cpdata via /var/crash/MLP
ShareDir=/var/crash/MLP/${USER}
export SCRIPT_DIR=${ShareDir}
mkdir -p ${ShareDir}

export CPSCRIPT="cpdata_decomp8K.sh"
#export CPSCRIPT="cpdata_decomp.sh"
#TarFile="opt.tgz"
#TarFile="opt.tar.gz"
#TarFile="opt_20201021b.tgz"
#TarFile="opt_20201022_26bTF.tgz"
#TarFile="opt_20201022_27aMPI4PY.tgz"
TarFile="opt_20201022_27aMPI4PY_2nd.tgz"
JobID=${PJM_SUBJOBID}
CheckFilePrefix="opt_copy_finished_"
CheckFile=${CheckFilePrefix}${JobID}
export OPT_PATH=${ShareDir}/opt
#. ./setenv_tmp
. ./setenv_tmp_27a

Host=`hostname`
HostPost=${Host: -1}
if [ ${HostPost} = 'b' ];then

    chmod 777 -R ${ShareDir}
    umask 000 ${ShareDir}

    echo "I am BIO (`hostname`)"
    rm -f ${ShareDir}/${CheckFilePrefix}*
    rm -rf ${ShareDir}/opt* :
    rm -f ${ShareDir}/cpdata.sh :

    #time -p cp ${COSMOFLOW_BASE}/${TarFile} ${ShareDir}/${TarFile}
    # 4-volume distributed
    SRC_FILE=/vol000$(($RANK/16%4+1))/input_MLPerf/u93182/${TarFile}
    time -p cp ${SRC_FILE} ${ShareDir}/${TarFile}
    chmod 666 ${ShareDir}/${TarFile}

    time -p tar -I pigz -xf ${ShareDir}/${TarFile} -C ${ShareDir}
    cp ./${CPSCRIPT} ${ShareDir}/cpdata.sh

    touch ${ShareDir}/${CheckFile}
else
    pushd ${ShareDir}
    while [ ! -e ${CheckFile} ]
    do
        sleep 10
    done
    popd
fi

TMP_DEST=${LocalDataDir}
mkdir -p ${TMP_DEST}
chmod 777 -R ${TMP_DEST}
umask 000 ${TMP_DEST}

LIBTCMALLOC_DIR="/vol0001/apps/oss/PyTorch-1.5.0/lib"

. "$ParameterFile"

LogDir=${LOGDIR}

export PLE_MPI_STD_EMPTYFILE="off"
export OMP_PROC_BIND=false
#export OMP_NUM_THREADS=28
#export HOROVOD_MPI_THREADS_DISABLE=1
export TF_NUM_INTEROP_THREADS=1
export TF_MKL_OPTIMIZE_PRIMITIVE_MEMUSE=false
export LD_PRELOAD=${LIBTCMALLOC_DIR}/libtcmalloc.so
export TMPDIR=$TMP_DEST #for python tempfile

#env > $LogDir/env_${RANK}

#export MKLDNN_JIT_DUMP=1
#export MKLDNN_VERBOSE=2

if [ $RANK -eq "0" ]; then
    export DNNL_VERBOSE=$dnnlverbose
    env > ${LogDir}/env_${RANK}
fi

which numactl
which python

#time -p ${ShareDir}/cpdata.sh $DataDir $TMP_DEST 'xz' $SIZE ${GroupSize}

time -p numactl --cpunodebind 4-7 --membind 4-7 \
    python "train.py" "${PARAMS[@]}"

rm -rf ${TMP_DEST}
rm -rf ${ShareDir}

date

exit
