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

# Note to adapt number of training/validation samples when changing max. rank number  
# (n_train=256 simulates 1024 Cosmoflow nodes, # training files = 4 * # validation files)
log_max_ranks=0
n_train_per_rank=2048
n_valid_per_rank=512

inter_threads_range=(0 1 2 3 4) # (2)
intra_threads_range=(0 6 12 24 36 48) # (12)
set +x


# Weak scaling on Piz Daint (simulates 2048 Cosmoflow nodes)
for log_n_ranks in $(seq 0 ${log_max_ranks}); do
  n_ranks=$((2**log_n_ranks))
  for inter_threads in ${inter_threads_range[@]}; do
    for intra_threads in ${intra_threads_range[@]}; do
      set -x
      sbatch -N ${n_ranks} ${sbatch_script} --data-benchmark  \
          --data-dir ${data_dir} \
          --output-dir "results/data_benchmark/${output_dir_postfix}/gpu-n${n_ranks}-inter${inter_threads}-intra${intra_threads}" \
           --n-train $((${n_train_per_rank} * ${n_ranks})) --n-valid $((${n_valid_per_rank} * ${n_ranks})) --n-epochs 5 \
           --inter-threads ${inter_threads} --intra-threads ${intra_threads} \
           configs/cosmo.yaml
      set +x
    done
  done
done


# Strong scaling also of interest
