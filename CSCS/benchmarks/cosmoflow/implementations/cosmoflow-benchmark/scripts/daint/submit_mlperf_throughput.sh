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

# Node number/global batch size range/weak scaling parameters
if [ ${testing} ] ; then 
    min_log_n_ranks=1 
    max_log_n_ranks=1
    log_global_batch_size_range=(1)
    n_epochs=4

    weak_scaling=true
    # Weak scaling (default) local dataset size
    weak_n_train_per_rank=256 
    weak_n_valid_per_rank=64
    # Strong scaling global dataset size
    strong_n_train=$(( 256*2**${max_log_n_ranks} ))
    strong_n_valid=$((  64*2**${max_log_n_ranks} ))
else # full-scale benchmarking
    min_log_n_ranks=7
    max_log_n_ranks=10
    log_global_batch_size_range="$(seq ${max_log_n_ranks} -1 ${min_log_n_ranks})"
    n_epochs=4

    weak_scaling=false
    # Weak scaling local dataset size
    weak_n_train_per_rank=$(( 2**(18 - ${max_log_n_ranks}) ))
    weak_n_valid_per_rank=$(( 2**(16 - ${max_log_n_ranks}) ))
    # Strong scaling (default) global dataset size
    strong_n_train=$(( 2**(18 - ${max_log_n_ranks}) ))    
    strong_n_valid=$(( 2**(16 - ${max_log_n_ranks}) ))    
fi

# Horovod/NCCL options
export HOROVOD_CYCLE_TIME=0.001
export HOROVOD_FUSION_THRESHOLD=512
export HOROVOD_HIERARCHICAL_ALLREDUCE=1
export NCCL_ALGO=Tree
# Sarus/TF 2.3: remove this due to bug in XLA/ptxas
export TF_XLA_FLAGS="--tf_xla_auto_jit=2" 
export TF_GPU_THREAD_MODE=gpu_private
export NCCL_DEBUG=WARN
set +x

# Shell helpers
max() {
   [ "$1" -gt "$2" ] && echo $1 || echo $2
}

min() {
   [ "$1" -lt "$2" ] && echo $1 || echo $2
}

# Strong scaling throughput
for log_global_batch_size in ${log_global_batch_size_range[@]}; do
    log_n_ranks_ubound=$( min $max_log_n_ranks $log_global_batch_size )
    log_n_ranks_lbound=$( max $min_log_n_ranks $(( log_global_batch_size - 3 )) ) # OOM error for local batch size > 8 on P100
    for log_n_ranks in $( seq $( min $max_log_n_ranks $log_global_batch_size ) -1 $( max $min_log_n_ranks $(( log_global_batch_size - 3 )) ) ); do
        n_ranks=$(( 2**${log_n_ranks} ))
        local_batch_size=$(( 2**(${log_global_batch_size}-${log_n_ranks}) ))

        if [ ${weak_scaling} ] ; then
            dataset_extra_opts="--n-train $((${weak_n_train_per_rank} * ${n_ranks})) --n-valid $((${weak_n_valid_per_rank} * ${n_ranks}))"
        else # strong scaling
            dataset_extra_opts="--n-train ${strong_n_train} --n-valid ${strong_n_valid}"
        fi

        #if [ "${local_batch_size}" -le 8 ]; then # OOM error for local batch size > 8 on P100
        set -x
        sbatch -N ${n_ranks}  "${sbatch_script}"  \
          --data-dir ${data_dir} ${dataset_extra_opts} \
          --batch-size ${local_batch_size} --n-epochs ${n_epochs} \
          --output-dir "results/mlperf_throughput/${output_dir_postfix}/gpu-n${n_ranks}_batch${local_batch_size}" \
          configs/cosmo.yaml
        set +x
        #fi
    done
done
