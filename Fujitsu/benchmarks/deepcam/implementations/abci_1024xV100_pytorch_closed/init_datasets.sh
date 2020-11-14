#!/bin/bash

group_id="gca50115"
num_nodes=16
runtime="2:00:00"

cd ../implementation_abci/reformat
qsub -g ${group_id} -l rt_F=${num_nodes} -l h_rt=${runtime} -j y -cwd ./run_reformat.sh
