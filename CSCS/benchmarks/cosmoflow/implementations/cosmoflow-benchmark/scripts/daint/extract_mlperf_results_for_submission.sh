#!/bin/bash

set -euo pipefail

# Parameters are derived from folder structure in results directory
echo "File path parameters:"
echo "mlperf_run is the experiment type"
echo "$1 is the timestamp_hostname label"

input_dir="results/mlperf_run/$1"
output_dir="results/mlperf_run/$1-results"

echo "Extracting mlperf.logs from ${input_dir} to ${output_dir} for submission."

mkdir -p ${output_dir}

for d in $(ls -d ${input_dir}/* | grep -v /logs); do
    cp  "${d}/mlperf.log" "${output_dir}/result_$(basename ${d}).txt"
done
