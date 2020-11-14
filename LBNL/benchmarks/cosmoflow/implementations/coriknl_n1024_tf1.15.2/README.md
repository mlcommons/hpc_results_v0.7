# CosmoFlow benchmark, coriknl\_n1024\_tf1.15.2

Open division submission on 1024 Cori KNL nodes.

To run:

```bash
sbatch --array=0-10%1 -N 512 -q regular -t 12:00:00 \
    scripts/train_cori_shifter.sh configs/cosmo_runs_knl_open.yaml \
    --kmp-blocktime 1 --omp-num-threads 32 --intra-threads 32 --inter-threads 2
```
