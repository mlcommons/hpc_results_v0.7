# CosmoFlow benchmark, coriknl\_n512\_tf1.15.2

To run:

```bash
sbatch --array=0-10%1 -N 512 -q regular -t 12:00:00 \
    scripts/train_cori_shifter.sh configs/cosmo_runs_knl.yaml \
    --kmp-blocktime 1 --omp-num-threads 32 --intra-threads 32 --inter-threads 2
```
