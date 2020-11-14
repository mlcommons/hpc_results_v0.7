#!/bin/bash

RANK=$PMIX_RANK
SIZE=$PJM_NODE

echo `date` $RANK / $SIZE Job start

. ${HOME}/COSMOFLOW_27b/setenv

LIBTCMALLOC_DIR=${HOME}/PyTorch-1.5.0/lib

export PLE_MPI_STD_EMPTYFILE="off"

LogDir=$1

unset KMP_AFFINITY
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=40
export HOROVOD_MPI_THREADS_DISABLE=1
export TF_NUM_INTEROP_THREADS=1
export TF_MKL_OPTIMIZE_PRIMITIVE_MEMUSE=false
export LD_PRELOAD=${LIBTCMALLOC_DIR}/libtcmalloc.so

env > $LogDir/env_${OMPI_RANK}

#export MKLDNN_JIT_DUMP=1

LOGFILE="${LogDir}/${RANK}_cosmo_train_tf220_fugaku_p${PJM_MPI_PROC}.log"
if [ $RANK -eq 0 ]; then
    free -h > ${LogDir}/result_free
    df > ${LogDir}/result_df
    env > ${LogDir}/result_env_${RANK}
    #export MKLDNN_VERBOSE=2
fi

echo `date` " python start"
numactl --cpunodebind 4-7 --membind 4-7 \
    python -u train.py \
            -d \
            --data-dir ${COSMOFLOW_BASE}/data \
            --output-dir $LogDir \
            --batch-size 1 \
            --n-train 1024 \
            --n-epochs 1 \
    > ${LogDir}/stderrout.${RANK} 2>&1

echo `date` $RANK Job end
