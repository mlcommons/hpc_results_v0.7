#!/bin/bash

cd ../implementation_abci_fj/reformat
mpirun -n 16 ./reformat_mpi.sh 256 /path/to/original/dataset/All-Hist
