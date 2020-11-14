#!/bin/bash

set -ex
timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
output_dir_postfix="${timestamp}_${HOSTNAME}"

if [ "$#" -lt 1 ] || [[ $1 -ne 128 ]] && [[ $1 -ne 256 ]]; then
    echo "Error: Expecting n_ranks in [128, 256] as a first argument." >&2; exit 1
else
    n_ranks="$1"
fi

if [ "$#" -lt 2 ]; then
    use_container=false
else
    if [ "$2" = "sarus" ]; then
        use_container=true
    elif [ "$2" = "module" ]; then
        use_container=false
    else
        echo "Use either 'sarus' or 'module' to run the container/module-version"
    fi
fi

if [ "$#" -lt 3 ]; then
    testing=false
else
    if [ "$3" = "testing" ]; then
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

n_runs=10

# n_ranks set as command-line parameter
local_batch_size=2
global_batch_size=$(( ${n_ranks} * ${local_batch_size} ))
config_file="configs/cosmo-${global_batch_size}-sub.yaml"

if [ ${testing} ] ; then
    # testing at small scale
    n_ranks=2
    testing_n_train_per_rank=256
    testing_n_valid_per_rank=64
    testing_n_epochs=2
fi

RESERVATION="" # "--reservation=<slurm-reservation-name>"

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

for instance in $(seq 1 ${n_runs}); do
    if [ ${testing} ]; then
        testing_extra_opts="--n-train $((${testing_n_train_per_rank} * ${n_ranks})) --n-valid $((${testing_n_valid_per_rank} * ${n_ranks})) --n-epochs        ${testing_n_epochs}"
    else
        testing_extra_opts=""
    fi

  set -x
  sbatch ${RESERVATION} -N ${n_ranks}  ${sbatch_script}  \
      --data-dir ${data_dir} ${testing_extra_opts} \
      --output-dir "results/mlperf_run/${output_dir_postfix}/gpu-gb${global_batch_size}-n${n_ranks}-sub${instance}" \
      ${config_file}
  set +x
done
