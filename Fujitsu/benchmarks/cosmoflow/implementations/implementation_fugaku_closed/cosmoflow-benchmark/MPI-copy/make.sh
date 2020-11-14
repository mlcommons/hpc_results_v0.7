#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=dvsys-huge"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM --mpi "max-proc-per-node=1" 
#PJM -j
#PJM -S

export PLE_MPI_STD_EMPTYFILE="off"

module load lang/tcsds-1.2.27b

mpifcc -o mpi_bcast mpi_bcast.c -Kfast
