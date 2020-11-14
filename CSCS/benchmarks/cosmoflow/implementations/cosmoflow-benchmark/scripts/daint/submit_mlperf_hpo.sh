#!/bin/bash

set -ex
timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
output_dir_postfix="${timestamp}_${HOSTNAME}"

if [ "$#" -lt 1 ]; then
    use_container=false
else
    if [ "$1" = "sarus" ]; then
        use_container=true
    elif [ "$1" = "module" ]; then
        use_container=false
    else
        echo "Use either 'sarus' or 'module' to run the container/module-version"
    fi  
fi

if [ "$#" -lt 2 ]; then
    testing=false
else
    if [ "$2" = "testing" ]; then
        testing=true
    else
        testing=false
    fi
fi

# Data set & sbatch script
if [ ${use_container} ] ; then
    # Sarus data_dir
    data_dir=/root/mlperf/data/cosmoflow/cosmoUniverse_2019_05_4parE_tf
    sbatch_script=scripts/daint/throughput_sarus.sh
else
    # Module data_dir
    data_dir=/scratch/snx3000/lukasd/mlperf/data/cosmoflow/cosmoUniverse_2019_05_4parE_tf
    sbatch_script=scripts/daint/throughput_daint.sh
fi

# Node number/global batch size range
if [ ${testing} ] ; then
    n_ranks=2
    batch_size=4
    global_batch_size=512 # 1024
    base_batch_range=(96 128 160) # (192 256 384)

    testing_n_train_per_rank=256
    testing_n_valid_per_rank=64
    testing_n_epochs=4
else
    n_ranks=128 # 256
    batch_size=4 # 2
    global_batch_size=$(( ${n_ranks} * ${batch_size} ))
    base_batch_range=(96 128 160) # (192 256 384)
fi

# Horovod/NCCL options
export HOROVOD_CYCLE_TIME=0.001
export HOROVOD_FUSION_THRESHOLD=512
export HOROVOD_HIERARCHICAL_ALLREDUCE=1
export NCCL_ALGO=Tree
# Sarus/TF 2.3: remove this due to bug in XLA/ptxas
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
export TF_GPU_THREAD_MODE=gpu_private
set +x

for base_batch_size in ${base_batch_range[@]}; do
    if [ ${testing} ]; then
        testing_extra_opts="--n-train $((${testing_n_train_per_rank} * ${n_ranks})) --n-valid $((${testing_n_valid_per_rank} * ${n_ranks})) --n-epochs ${testing_n_epochs}"
    else
        testing_extra_opts=""
    fi
    set -x
    sbatch -N ${n_ranks} ${sbatch_script} \
        --data-dir ${data_dir}  ${testing_extra_opts} --batch-size ${batch_size} \
        --output-dir "results/mlperf_hpo/${output_dir_postfix}/gpu-gb${global_batch_size}-bb${base_batch_size}" \
        configs/cosmo-${global_batch_size}-bb${base_batch_size}.yaml 
    set +x
done

