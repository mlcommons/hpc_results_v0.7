# DeepCAM benchmark, corigpu\_n64\_pt1.6.0

Closed division submission on Cori GPU, 8 nodes, 64 GPUs.

To run:

```bash
cd src/deepCam/run_scripts
sbatch --array="1-5%1" -N 8 -t 8:00:00 run_training_cgpu.sh
```
