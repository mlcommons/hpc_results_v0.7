### CSCS Cosmoflow submission

The ML-Perf submission results can be obtained on Piz Daint by loading the modules `daint-gpu` and `Horovod/0.19.1-CrayGNU-20.08-tf-2.2.0`, configuring the environment appropriately and submitting an sbatch job. All of this is done in

```
scripts/daint/submit_mlperf_run.sh 128 module
```

for 128 nodes and with

```
scripts/daint/submit_mlperf_run.sh 256 module
```

for 256 nodes.

Alternatively, the results can be reproduced with the Docker container runtime [`Sarus`](https://link.springer.com/chapter/10.1007/978-3-030-34356-9_5). For this, build a Docker images outside of Piz Daint with

```
cd builds
docker build -f Dockerfile.tf22.gpu_daint -t cosmoflow_gpu_daint .
```

and save it with `docker save cosmoflow_gpu_daint -o cosmoflow_gpu_daint.tar`. To make it available to `Sarus`, it is copied to Piz Daint and loaded with `sarus load cosmoflow_gpu_daint.tar cosmoflow_gpu_daint`.

The training data for cosmoflow is expected to reside under `../data/cosmoflow` w.r.t. this repository. To avoid multiple data copies on the same machine when testing different versions of the Cosmoflow code, use e.g. symbolic links to the `data` directory (on Piz Daint to `/scratch/snx3000/lukasd/mlperf/data`).

The submission results can then be reproduced `scripts/daint/submit_mlperf_run.sh <XX> sarus`, with `<XX>` either 128 or 256 (i.e. by replacing `module` in the above commands with `sarus`).

