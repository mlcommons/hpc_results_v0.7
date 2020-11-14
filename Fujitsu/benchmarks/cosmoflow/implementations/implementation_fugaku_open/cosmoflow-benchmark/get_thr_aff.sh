#!/bin/sh
proc_id=$1

echo "ProcessID = ${proc_id}"
thread_list=`ps -eTo pid,tid,comm | grep ${proc_id} | grep -v grep | awk '{print $2}' `
for thread_id in ${thread_list}
do
    taskset -p ${thread_id}
done
echo "========"

exit 0
