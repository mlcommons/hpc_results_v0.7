#!/bin/sh

if [ $# -lt 3 ]; then
  echo "$0 [data_path] [local_dir] [local_rank] [debug (default: off)]"
  echo "data staging exit."
  exit 1
fi

data_path=$1
local_dir=$2
local_rank=$3
if [ $# -gt 3 ]; then
  debug=$4
else
  debug=""
fi

# 0: no data staging if data is stored in compute nodes
# 1: remove existing data in compute nodes (if any) and do data staging
force_data_staging=0

nprocs=${OMPI_COMM_WORLD_SIZE}
rank=${OMPI_COMM_WORLD_RANK}
#local_rank=${OMPI_COMM_WORLD_LOCAL_RANK}
local_size=${OMPI_COMM_WORLD_LOCAL_SIZE}
num_nodes=$((${nprocs} / ${local_size}))
node_id=$((${rank} / ${local_size}))

if [ ${local_rank} -ne 0 ]; then
  exit 0
fi

if [ ${force_data_staging} -gt 0 ]; then
  echo "remove data in local disk"
  rm -rf ${SGE_LOCALDIR}/*
fi
mkdir -p ${local_dir}

if [ ${rank} -eq 0 ]; then
  echo "staging stats data"
fi
if [ ${local_rank} -eq 0 ]; then
  cp ${data_path}/stats.h5 ${local_dir}
fi

min_num_files=64
if [ ${num_nodes} -lt ${min_num_files} ]; then
  num_files_per_node=$((${min_num_files} / ${num_nodes}))
  start_idx=$((${node_id} * ${num_files_per_node}))
  end_idx=$((${start_idx} + ${num_files_per_node} - 1))
  num_nodes=${min_num_files}
else
  num_files_per_node=1
  start_idx=${node_id}
  end_idx=${node_id}
fi

if [ ${local_rank} -eq 0 ]; then
  train_local_dir=${local_dir}/train
  if [ ! -d ${train_local_dir} ]; then
    mkdir -p ${train_local_dir}
    if [ ! -z ${debug} ]; then
      if [ ${rank} -eq 0 ]; then
        echo "copy and extract data"
      fi
      echo "tar xf /groups1/gca50115/datasets/deepcam/reformatted/64/train_64_$((${rank} + 1)).tar -C ${train_local_dir}"
      tar xf /groups1/gca50115/datasets/deepcam/reformatted/64/train_64_$((${rank} + 1)).tar -C ${train_local_dir}
    else
      if [ ${rank} -eq 0 ]; then
        echo "staging train data"
      fi
      for file_idx in `seq ${start_idx} ${end_idx}`; do
        train_data=${data_path}/${num_nodes}/train_${num_nodes}_$((${file_idx} + 1)).tar
        tar xf ${train_data} -C ${train_local_dir}
      done
    fi
  fi  

  validation_local_dir=${local_dir}/validation
  if [ ! -d ${validation_local_dir} ]; then
    mkdir -p ${validation_local_dir}
    if [ ! -z ${debug} ]; then
      echo "mkdir -p ${validation_local_dir}"
      mkdir -p ${validation_local_dir}
      echo "tar xf /groups1/gca50115/datasets/deepcam/reformatted/64/validation_64_$((${rank} + 1)).tar -C ${validation_local_dir}"
      tar xf /groups1/gca50115/datasets/deepcam/reformatted/64/validation_64_$((${rank} + 1)).tar -C ${validation_local_dir}
    else
      if [ ${rank} -eq 0 ]; then
        echo "staging validation data"
      fi
      for file_idx in `seq ${start_idx} ${end_idx}`; do
        validation_data=${data_path}/${num_nodes}/validation_${num_nodes}_$((${file_idx} + 1)).tar
        tar xf ${validation_data} -C ${validation_local_dir}
      done
    fi
  fi
  if [ ! -z ${debug} ]; then
    num_local_train_files=`ls ${local_dir}/train | wc -l`
    num_local_validation_files=`ls ${local_dir}/validation | wc -l`
    echo "rank: ${rank}, num_local_train_files: ${num_local_train_files}, num_local_validation_files: ${num_local_validation_files}"
  fi
fi
