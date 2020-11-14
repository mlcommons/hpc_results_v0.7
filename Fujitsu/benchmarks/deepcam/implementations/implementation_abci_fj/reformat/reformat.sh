#!/bin/bash

if [ $# -lt 4 ]; then
  echo "$0 [num_nodes] [start_idx] [end_idx] [data_path]"
  exit
fi

num_nodes=$1
start_idx=$2
end_idx=$3
orig_data_path=$4

mkdir -p ${num_nodes}
cd ${num_nodes}

generate_train_files=false
generate_validation_files=true


if [ "${generate_train_files}" ]; then
  num_train_files=`ls ${orig_data_path}/train | grep data | wc -l`
  num_train_files_per_node=$((${num_train_files} / ${num_nodes}))
  num_train_files_surplus=$((${num_train_files} % ${num_nodes}))
  echo "num train files: $num_train_files"
  echo "num train files per node: ${num_train_files_per_node}"
  echo "num train files surplus: ${num_train_files_surplus}"
  
  if [ ${num_train_files_surplus} -gt 0 ]; then
    num_train_files_per_node=$((${num_train_files_per_node} + 1))
  fi
  
  train_files_head=1
  echo "train files head: ${train_files_head}"
  
  for i in `seq 1 ${num_nodes}`; do
    cur_train_idx=train_${num_nodes}_${i}
  
    if [ ${i} -eq $((${num_train_files_surplus} + 1)) ]; then
      num_train_files_per_node=$((${num_train_files_per_node} - 1))
    fi
    echo "for node ${i}: num train files per node: ${num_train_files_per_node}"
    echo "for node ${i}: train_files from ${train_files_head} to $((${train_files_head} + ${num_train_files_per_node} - 1))"
  
    if [ ${i} -lt ${start_idx} ] || [ ${i} -gt ${end_idx} ]; then
      echo "skip"
    elif [ -f ${cur_train_idx}.tar ]; then
      echo "${cur_train_idx}.tar exists. skip."
    else
      echo "tar cf ${cur_train_idx}.tar -C ${orig_data_path}/train [files]"
      train_files=`ls ${orig_data_path}/train | grep data | head -n $((${train_files_head} + ${num_train_files_per_node} - 1)) | tail -n ${num_train_files_per_node}`
      tar cf ${cur_train_idx}.tar -C ${orig_data_path}/train ${train_files}
    fi
  
    train_files_head=$((${train_files_head} + ${num_train_files_per_node} ))
  done
fi

if [ "${generate_validation_files}" ]; then
  num_validation_files=`ls ${orig_data_path}/validation | grep data | wc -l`
  num_validation_files_per_node=$((${num_validation_files} / ${num_nodes}))
  num_validation_files_surplus=$((${num_validation_files} % ${num_nodes}))
  echo "num validation files: $num_validation_files"
  echo "num validation files per node: ${num_validation_files_per_node}"
  echo "num validation files surplus: ${num_validation_files_surplus}"
  
  if [ ${num_validation_files_surplus} -gt 0 ]; then
    num_validation_files_per_node=$((${num_validation_files_per_node} + 1))
  fi
  
  validation_files_head=1
  echo "validation files head: ${validation_files_head}"
  
  for i in `seq 1 ${num_nodes}`; do
    cur_validation_idx=validation_${num_nodes}_${i}
  
    if [ ${i} -eq $((${num_validation_files_surplus} + 1)) ]; then
      num_validation_files_per_node=$((${num_validation_files_per_node} - 1))
    fi
    echo "for node ${i}: num validation files per node: ${num_validation_files_per_node}"
    echo "for node ${i}: validation_files from ${validation_files_head} to $((${validation_files_head} + ${num_validation_files_per_node} - 1))"
  
    if [ ${i} -lt ${start_idx} ] || [ ${i} -gt ${end_idx} ]; then
      echo "skip"
    elif [ -f ${cur_validation_idx}.tar ]; then
      echo "${cur_validation_idx}.tar exists. skip."
    else
      echo "tar cf ${cur_validation_idx}.tar -C ${orig_data_path}/validation [files]"
      validation_files=`ls ${orig_data_path}/validation | grep data | head -n $((${validation_files_head} + ${num_validation_files_per_node} - 1)) | tail -n ${num_validation_files_per_node}`
      tar cf ${cur_validation_idx}.tar -C ${orig_data_path}/validation ${validation_files}
    fi
  
    validation_files_head=$((${validation_files_head} + ${num_validation_files_per_node} ))
  done
fi

cd ..
echo "done reformat."
