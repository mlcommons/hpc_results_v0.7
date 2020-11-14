#!/bin/bash

set -ex

timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
output_dir_postfix="${timestamp}_${HOSTNAME}"

# Scaling locally
for log_n_ranks in $(seq 0 0); do
    n_ranks=$((2**log_n_ranks))
    docker run --rm \
               -v $(pwd)/../data:/root/mlperf/data \
               -v $(pwd):/root/mlperf/cosmoflow-benchmark \
               -w /root/mlperf/cosmoflow-benchmark \
               cosmoflow_gpu_daint \
    mpiexec -np ${n_ranks} python train.py --distributed \
        --output-dir "results/weak_scaling/${output_dir_postfix}/scaling-n${n_ranks}" \
        --n-train $((2 * ${n_ranks})) --n-valid $((2 * ${n_ranks})) --batch-size 1 --n-epochs 8 \
        --conv-size 2 --n-conv-layers 1 --fc1-size 16 --fc2-size 8 \
        --verbose configs/cosmo_local.yaml
done

# Scaling locally with dummy data
for log_n_ranks in $(seq 0 0); do
    n_ranks=$((2**log_n_ranks))
    docker run --rm \
        -v $(pwd)/../data:/root/mlperf/data \
        -v $(pwd):/root/mlperf/cosmoflow-benchmark \
        -w /root/mlperf/cosmoflow-benchmark \
        cosmoflow_gpu_daint \
    mpiexec -np ${n_ranks}  python train.py --distributed \
        --output-dir "results/weak_scaling/${output_dir_postfix}/scaling-dummy-n${n_ranks}" \
        --n-train $((2 * ${n_ranks})) --n-valid $((2 * ${n_ranks})) --batch-size 1 --n-epochs 8 \
        --conv-size 2 --n-conv-layers 1 --fc1-size 16 --fc2-size 8 \
        --verbose configs/cosmo_dummy_local.yaml
done
