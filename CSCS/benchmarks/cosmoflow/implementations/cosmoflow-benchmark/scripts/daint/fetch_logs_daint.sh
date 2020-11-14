#!/bin/bash

set -euo pipefail

# Parameters are derived from folder structure in results directory
echo "$1 is the experiment type (weak_scaling, data_benchmark, mlperf_run, etc.)"
echo "$2 is the timestamp_hostname label"

mkdir -p results/$1/$2
scp -r "daint:/scratch/snx3000/lukasd/mlperf/cosmoflow-benchmark/results/$1/$2" results/$1/

mkdir -p results/$1/$2/logs/
logfiles_list=$(logfiles=""; while read -r line; do logfiles+=,$line; done < <(ssh daint101 "cd /scratch/snx3000/lukasd/mlperf/cosmoflow-benchmark/logs && grep -l $2 *"); echo "${logfiles:1}")

set -x
scp -r "daint:/scratch/snx3000/lukasd/mlperf/cosmoflow-benchmark/logs/{${logfiles_list}}" results/$1/$2/logs/
set +x
