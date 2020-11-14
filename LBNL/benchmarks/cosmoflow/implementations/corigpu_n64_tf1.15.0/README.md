# CosmoFlow benchmark, corigpu\_n64\_tf1.15.0

Closed division submission on Cori GPU, 8 nodes, 64 GPUs.

To run:

```bash
sbatch --array="1-10%1" -N 8 -t 8:00:00 \
    scripts/train_cgpu.sh configs/cosmo_runs_gpu.yaml \
    --stage-dir /tmp/cosmoUniverse_2019_05_4parE_tf
```
