#!/bin/bash

mpirun -n 8 --tag-output python examples/cosmo.py  |& tee $1
