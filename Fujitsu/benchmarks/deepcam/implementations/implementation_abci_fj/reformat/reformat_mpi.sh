#!/bin/bash

if [ $# -lt 2 ]; then
  echo "$0 [num_nodes] [data_path]"
  exit
fi

num_nodes=$1
data_path=$2

rank=${OMPI_COMM_WORLD_RANK}
size=${OMPI_COMM_WORLD_SIZE}

num_files_per_proc=$((${num_nodes} / ${size}))

if [ ${size} -gt ${num_nodes} ]; then
  echo "size ${size} is larger than num_nodes ${num_nodes}. exit."
  exit
fi

if [ $((${num_nodes} % ${size})) -ne 0 ]; then
  echo "num_nodes ${num_nodes} is not divisible by size ${size}. exit."
  exit
fi

start_idx=$((${num_files_per_proc} * ${rank} + 1))
end_idx=$((${start_idx} + ${num_files_per_proc} - 1))

./reformat.sh ${num_nodes} ${start_idx} ${end_idx} ${data_path}

echo "rank ${rank}: done."
