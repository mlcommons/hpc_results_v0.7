#!/bin/bash

group_id=gca50115

cd ../implementation_abci_old/preproc
qsub -g ${group_id} ./pack.sh
