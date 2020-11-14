#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=dvsys-huge"
#PJM -L elapse=01:00:00
#PJM -L "node=16"
#PJM --mpi "max-proc-per-node=1" 
#PJM -j
#PJM -S


module load lang/tcsds-1.2.27b
Nnodes=`pjshowip | wc | awk '{print $1}'`

mpiexec -n ${Nnodes} -of remove-file.log ./remove-tmp-dir.sh
